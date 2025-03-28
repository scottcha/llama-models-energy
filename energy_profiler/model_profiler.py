import torch
import torch.nn as nn
import sys
import os
import copy
from types import MethodType

# Add parent directory to the path to be able to import from energy_profiler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from energy_profiler.power_profiler import PowerProfiler
from energy_profiler.model_loader import load_llama_model

class EnergyProfiledModule(nn.Module):
    """Wrapper around a module that can be enabled/disabled for energy profiling."""
    
    def __init__(self, original_module, name=None, enabled=True):
        """Initialize the energy profiled module.
        
        Args:
            original_module: The original module to wrap
            name: Name identifier for the module
            enabled: Whether the module should be enabled
        """
        super().__init__()
        self.original_module = original_module
        self.name = name
        self.enabled = enabled
        
        # Maintain the original forward signature
        self.forward.__func__.__signature__ = original_module.forward.__signature__
    
    def forward(self, *args, **kwargs):
        """Forward pass that either calls the original module or returns zeros.
        
        If enabled, performs the original computation.
        If disabled, returns a tensor of zeros with the same shape as the original output.
        """
        if not self.enabled:
            # Compute a reference output to get the shape
            with torch.no_grad():
                # Store original state
                original_state = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                
                # Compute reference output
                reference_output = self.original_module(*args, **kwargs)
                
                # Create zeros tensor with the same shape
                if isinstance(reference_output, tuple):
                    result = tuple(torch.zeros_like(x) for x in reference_output)
                else:
                    result = torch.zeros_like(reference_output)
                
                # Restore state
                torch.set_grad_enabled(original_state)
            
            return result
        else:
            # Normal operation - call the original module
            return self.original_module(*args, **kwargs)
    
    def __repr__(self):
        """String representation of the module."""
        status = "enabled" if self.enabled else "disabled"
        return f"EnergyProfiledModule({self.name}, {status})"


class LlamaEnergyProfiler:
    """Wrapper to profile energy usage of Llama models with component control."""
    
    def __init__(self, llama_model_class, model_path, device="cuda:0"):
        """Initialize the energy profiler for a Llama model.
        
        Args:
            llama_model_class: The Llama model class to instantiate
            model_path: Path to the model weights
            device: Device to run the model on
        """
        self.llama_model_class = llama_model_class
        self.model_path = model_path
        self.device = device
        self.power_profiler = PowerProfiler()
        
        # Load the model
        self.model = self._load_model()
        
        # Original model for reference
        self.original_model = copy.deepcopy(self.model)
        
        # Track modified components
        self.modified_components = {}
    
    def _load_model(self):
        """Load the Llama model and prepare it for profiling."""
        print(f"Loading model from {self.model_path}")
        model = load_llama_model(self.llama_model_class, self.model_path, self.device)
        model.eval()  # Set to evaluation mode
        return model
    
    def disable_component(self, component_type, name_pattern=None):
        """Disable all components of a specified type in the model.
        
        Args:
            component_type: Class type of components to disable (e.g., FeedForward)
            name_pattern: Optional name pattern to match specific components
            
        Returns:
            List of disabled component names
        """
        disabled_components = []
        
        # Process all named modules
        for name, module in self.model.named_modules():
            if isinstance(module, component_type):
                if name_pattern is None or name_pattern in name:
                    # Skip if already modified
                    if name in self.modified_components:
                        continue
                    
                    # Find parent module and attribute name
                    parent_name, attr_name = self._get_parent_and_attr(name)
                    parent = self._get_module_by_name(self.model, parent_name)
                    
                    if parent is not None and hasattr(parent, attr_name):
                        # Create energyprofiled module
                        energy_module = EnergyProfiledModule(module, name=name, enabled=False)
                        
                        # Replace in parent
                        setattr(parent, attr_name, energy_module)
                        
                        # Track modification
                        self.modified_components[name] = {
                            "parent": parent_name,
                            "attr": attr_name,
                            "enabled": False,
                            "type": component_type.__name__
                        }
                        
                        disabled_components.append(name)
                        print(f"Disabled component: {name}")
        
        return disabled_components
    
    def enable_component(self, component_name):
        """Enable a previously disabled component.
        
        Args:
            component_name: Name of the component to enable
            
        Returns:
            True if successful, False otherwise
        """
        if component_name not in self.modified_components:
            print(f"Component {component_name} not found in modified components")
            return False
        
        info = self.modified_components[component_name]
        parent = self._get_module_by_name(self.model, info["parent"])
        
        if parent is not None and hasattr(parent, info["attr"]):
            module = getattr(parent, info["attr"])
            if isinstance(module, EnergyProfiledModule):
                module.enabled = True
                self.modified_components[component_name]["enabled"] = True
                print(f"Enabled component: {component_name}")
                return True
        
        return False
    
    def enable_all_components(self):
        """Enable all previously disabled components.
        
        Returns:
            Number of components enabled
        """
        count = 0
        for name in self.modified_components:
            if self.enable_component(name):
                count += 1
        
        return count
    
    def reset_model(self):
        """Reset the model to its original state."""
        self.model = copy.deepcopy(self.original_model)
        self.modified_components = {}
        print("Model reset to original state")
    
    def _get_parent_and_attr(self, name):
        """Get parent module name and attribute name from a full module name.
        
        Args:
            name: Full module name with dots
            
        Returns:
            Tuple of (parent_name, attribute_name)
        """
        parts = name.split('.')
        parent_name = '.'.join(parts[:-1])
        attr_name = parts[-1]
        return parent_name, attr_name
    
    def _get_module_by_name(self, model, name):
        """Get a module by its name.
        
        Args:
            model: Model to search in
            name: Module name with dots
            
        Returns:
            Module if found, None otherwise
        """
        if not name:
            return model
            
        for n, m in model.named_modules():
            if n == name:
                return m
        
        return None
    
    def profile_inference(self, input_text, tokenizer, max_tokens=50, with_components=None, without_components=None):
        """Profile energy consumption during inference.
        
        Args:
            input_text: Input text for inference
            tokenizer: Tokenizer to use
            max_tokens: Maximum number of tokens to generate
            with_components: List of component types to enable (others will be disabled)
            without_components: List of component types to disable (others will be enabled)
            
        Returns:
            Tuple of (output_text, energy_statistics)
        """
        # Configure components if specified
        if with_components is not None:
            # Disable all components first
            for component_type in self.modified_components.values():
                self.disable_component(eval(component_type["type"]))
            
            # Then enable only the specified ones
            for component_type in with_components:
                self.enable_component(component_type)
        
        if without_components is not None:
            # Enable all components first
            self.enable_all_components()
            
            # Then disable the specified ones
            for component_type in without_components:
                self.disable_component(component_type)
        
        # Prepare input
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Define inference function
        def inference_fn():
            with torch.no_grad():
                return self.model.generate(
                    inputs=inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9
                )
        
        # Measure energy during inference
        outputs, measurements = self.power_profiler.measure_operation(
            inference_fn, component="inference"
        )
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate energy
        energy_stats = self.power_profiler.calculate_energy(measurements)
        
        return output_text, energy_stats
    
    def compare_components_energy(self, input_text, tokenizer, component_types, max_tokens=50, num_runs=3):
        """Compare energy usage with different components enabled/disabled.
        
        Args:
            input_text: Input text for inference
            tokenizer: Tokenizer to use
            component_types: List of component types to compare
            max_tokens: Maximum number of tokens to generate
            num_runs: Number of runs for each configuration
            
        Returns:
            Dictionary with energy statistics for each configuration
        """
        results = {
            "baseline": {"energy": [], "power": [], "duration": []},
        }
        
        # Add entries for each component type
        for comp_type in component_types:
            results[f"without_{comp_type.__name__}"] = {"energy": [], "power": [], "duration": []}
        
        # Baseline runs (full model)
        print("Running baseline (all components enabled)...")
        for i in range(num_runs):
            self.reset_model()  # Ensure clean state
            _, energy_stats = self.profile_inference(input_text, tokenizer, max_tokens)
            results["baseline"]["energy"].append(energy_stats["energy_joules"])
            results["baseline"]["power"].append(energy_stats["avg_power_watts"])
            results["baseline"]["duration"].append(energy_stats["duration_seconds"])
            print(f"  Run {i+1}/{num_runs}: {energy_stats['energy_joules']:.3f} J, {energy_stats['avg_power_watts']:.2f} W")
        
        # Runs with specific components disabled
        for comp_type in component_types:
            print(f"Running without {comp_type.__name__}...")
            for i in range(num_runs):
                self.reset_model()  # Ensure clean state
                self.disable_component(comp_type)
                _, energy_stats = self.profile_inference(input_text, tokenizer, max_tokens)
                key = f"without_{comp_type.__name__}"
                results[key]["energy"].append(energy_stats["energy_joules"])
                results[key]["power"].append(energy_stats["avg_power_watts"])
                results[key]["duration"].append(energy_stats["duration_seconds"])
                print(f"  Run {i+1}/{num_runs}: {energy_stats['energy_joules']:.3f} J, {energy_stats['avg_power_watts']:.2f} W")
        
        # Calculate averages
        for config, data in results.items():
            if data["energy"]:
                data["avg_energy"] = sum(data["energy"]) / len(data["energy"])
                data["avg_power"] = sum(data["power"]) / len(data["power"])
                data["avg_duration"] = sum(data["duration"]) / len(data["duration"])
        
        # Calculate energy savings
        baseline_energy = results["baseline"]["avg_energy"]
        for config, data in results.items():
            if config != "baseline" and "avg_energy" in data:
                energy_diff = baseline_energy - data["avg_energy"]
                energy_diff_pct = (energy_diff / baseline_energy) * 100
                data["energy_saved"] = energy_diff
                data["energy_saved_pct"] = energy_diff_pct
        
        return results