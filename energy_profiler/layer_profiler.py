"""Module for detailed layer-by-layer energy profiling of Llama models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict

from energy_profiler.model_profiler import EnergyProfiledModule, LlamaEnergyProfiler

class LayerEnergyProfiler(LlamaEnergyProfiler):
    """Extended energy profiler for layer-by-layer analysis."""
    
    def __init__(self, llama_model_class, model_path, device="cuda:0"):
        """Initialize the layer-wise energy profiler."""
        super().__init__(llama_model_class, model_path, device)
        self.layer_map = self._build_layer_map()
        self.layer_measurements = {}
    
    def _build_layer_map(self):
        """Build a map of all layers in the model by type and position.
        
        Returns:
            Dictionary with layer types and their instances by position
        """
        layer_map = defaultdict(list)
        position_counters = defaultdict(int)
        
        # Process all named modules
        for name, module in self.model.named_modules():
            # Skip the root module
            if name == '':
                continue
            
            # Get module type
            module_type = type(module).__name__
            
            # Extract layer info
            layer_info = {
                'name': name,
                'type': module_type,
                'module': module,
                'position': position_counters[module_type]
            }
            
            # Add to map
            layer_map[module_type].append(layer_info)
            position_counters[module_type] += 1
        
        # Sort each type by position
        for module_type in layer_map:
            layer_map[module_type] = sorted(layer_map[module_type], key=lambda x: x['position'])
            print(f"Found {len(layer_map[module_type])} layers of type {module_type}")
        
        return layer_map
    
    def disable_layer_by_position(self, layer_type, position):
        """Disable a specific layer by type and position.
        
        Args:
            layer_type: Type of the layer (e.g., 'LlamaMLP')
            position: Position index of the layer (0-based)
            
        Returns:
            Name of the disabled layer if successful, None otherwise
        """
        if layer_type not in self.layer_map:
            print(f"Layer type {layer_type} not found in model")
            return None
            
        if position >= len(self.layer_map[layer_type]):
            print(f"Position {position} out of range for layer type {layer_type}")
            return None
        
        layer_info = self.layer_map[layer_type][position]
        layer_name = layer_info['name']
        
        # Skip if already modified
        if layer_name in self.modified_components:
            layer_info = self.modified_components[layer_name]
            module = self._get_module_by_name(self.model, layer_name)
            if isinstance(module, EnergyProfiledModule):
                module.enabled = False
                self.modified_components[layer_name]["enabled"] = False
                print(f"Disabled layer: {layer_name} (position {position})")
                return layer_name
        
        # Find parent module and attribute name
        parent_name, attr_name = self._get_parent_and_attr(layer_name)
        parent = self._get_module_by_name(self.model, parent_name)
        
        if parent is not None and hasattr(parent, attr_name):
            # Get original module
            original_module = getattr(parent, attr_name)
            
            # Create energy profiled module
            energy_module = EnergyProfiledModule(original_module, name=layer_name, enabled=False)
            
            # Replace in parent
            setattr(parent, attr_name, energy_module)
            
            # Track modification
            self.modified_components[layer_name] = {
                "parent": parent_name,
                "attr": attr_name,
                "enabled": False,
                "type": layer_type,
                "position": position
            }
            
            print(f"Disabled layer: {layer_name} (position {position})")
            return layer_name
        
        return None
    
    def profile_all_layers(self, layer_type, input_text, tokenizer, max_tokens=20, num_runs=3):
        """Profile energy consumption for each layer of a specific type.
        
        Args:
            layer_type: Type of layer to profile (e.g., 'LlamaMLP', 'LlamaAttention')
            input_text: Input text for inference
            tokenizer: Tokenizer to use
            max_tokens: Maximum number of tokens to process
            num_runs: Number of runs for each configuration
            
        Returns:
            Dictionary with energy statistics for each layer
        """
        if layer_type not in self.layer_map:
            raise ValueError(f"Layer type {layer_type} not found in model")
        
        num_layers = len(self.layer_map[layer_type])
        print(f"Profiling {num_layers} layers of type {layer_type}")
        
        # Baseline run with all layers enabled
        baseline_results = {
            "energy": [], "power": [], "duration": []
        }
        
        # Run baseline measurements
        print("Running baseline (all layers enabled)...")
        for i in range(num_runs):
            self.reset_model()  # Ensure clean state
            _, energy_stats = self.profile_inference(input_text, tokenizer, max_tokens)
            baseline_results["energy"].append(energy_stats["energy_joules"])
            baseline_results["power"].append(energy_stats["avg_power_watts"])
            baseline_results["duration"].append(energy_stats["duration_seconds"])
            print(f"  Baseline run {i+1}/{num_runs}: {energy_stats['energy_joules']:.3f} J")
        
        # Calculate baseline average
        baseline_results["avg_energy"] = sum(baseline_results["energy"]) / num_runs
        baseline_results["avg_power"] = sum(baseline_results["power"]) / num_runs
        baseline_results["avg_duration"] = sum(baseline_results["duration"]) / num_runs
        
        # Layer-by-layer measurements
        layer_results = {}
        
        # Store a copy of the baseline for comparison
        layer_results["baseline"] = baseline_results
        
        # Measure each layer individually
        for position in range(num_layers):
            layer_key = f"{layer_type}_{position}"
            layer_results[layer_key] = {"energy": [], "power": [], "duration": []}
            
            print(f"Profiling layer at position {position}/{num_layers-1}...")
            for i in range(num_runs):
                # Reset model to enable all layers
                self.reset_model()
                
                # Disable specific layer
                self.disable_layer_by_position(layer_type, position)
                
                # Run inference and measure
                _, energy_stats = self.profile_inference(input_text, tokenizer, max_tokens)
                
                # Store results
                layer_results[layer_key]["energy"].append(energy_stats["energy_joules"])
                layer_results[layer_key]["power"].append(energy_stats["avg_power_watts"])
                layer_results[layer_key]["duration"].append(energy_stats["duration_seconds"])
                print(f"  Run {i+1}/{num_runs}: {energy_stats['energy_joules']:.3f} J")
            
            # Calculate averages
            layer_results[layer_key]["avg_energy"] = sum(layer_results[layer_key]["energy"]) / num_runs
            layer_results[layer_key]["avg_power"] = sum(layer_results[layer_key]["power"]) / num_runs
            layer_results[layer_key]["avg_duration"] = sum(layer_results[layer_key]["duration"]) / num_runs
            
            # Calculate energy difference from baseline
            if baseline_results["avg_energy"] > 0:
                energy_diff = baseline_results["avg_energy"] - layer_results[layer_key]["avg_energy"]
                energy_pct = (energy_diff / baseline_results["avg_energy"]) * 100
                layer_results[layer_key]["energy_saved"] = energy_diff
                layer_results[layer_key]["energy_saved_pct"] = energy_pct
                layer_results[layer_key]["layer_energy_contribution"] = energy_diff
                layer_results[layer_key]["layer_energy_contribution_pct"] = energy_pct
            
        self.layer_measurements[layer_type] = layer_results
        return layer_results
    
    def visualize_layer_energy(self, layer_type, output_path=None):
        """Generate visualization of layer-wise energy consumption.
        
        Args:
            layer_type: Type of layer to visualize
            output_path: Path to save the visualization
        """
        if layer_type not in self.layer_measurements:
            raise ValueError(f"No measurements available for layer type {layer_type}")
        
        results = self.layer_measurements[layer_type]
        
        # Extract layer positions and energy contributions
        positions = []
        energy_contributions = []
        
        for key, data in results.items():
            if key != "baseline" and "layer_energy_contribution" in data:
                # Extract position from key (e.g., LlamaMLP_5 -> 5)
                position = int(key.split('_')[-1])
                positions.append(position)
                energy_contributions.append(data["layer_energy_contribution"])
        
        # Sort by position
        sorted_data = sorted(zip(positions, energy_contributions))
        positions = [x[0] for x in sorted_data]
        energy_contributions = [x[1] for x in sorted_data]
        
        # Total energy contribution of this layer type
        total_layer_energy = sum(energy_contributions)
        total_model_energy = results["baseline"]["avg_energy"]
        layer_type_pct = (total_layer_energy / total_model_energy) * 100
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Bar chart with absolute energy contributions
        bars = ax1.bar(positions, energy_contributions)
        
        # Add values on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{energy_contributions[i]:.2f} J',
                    ha='center', va='bottom', rotation=0)
        
        # Add labels and title
        ax1.set_xlabel('Layer Position')
        ax1.set_ylabel('Energy Contribution (Joules)')
        ax1.set_title(f'Energy Contribution by Layer Position ({layer_type})')
        
        # Add grid
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Calculate percentage of total model energy
        energy_pct = [100 * e / total_model_energy for e in energy_contributions]
        
        # Stacked bar showing percentage breakdown
        ax2.bar([0], [layer_type_pct], label=f'All {layer_type} Layers')
        ax2.bar([0], [100 - layer_type_pct], bottom=[layer_type_pct], 
                label='Rest of Model')
        
        # Add percentage labels
        ax2.text(0, layer_type_pct/2, f'{layer_type_pct:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold')
        ax2.text(0, layer_type_pct + (100-layer_type_pct)/2, f'{100-layer_type_pct:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Set labels and title
        ax2.set_title('Energy Distribution')
        ax2.set_ylabel('Percentage of Total Energy')
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Model Components'])
        ax2.legend()
        
        # Set the y-axis limit to 100%
        ax2.set_ylim(0, 100)
        
        # Add summary text
        plt.figtext(0.5, 0.01, 
                   f"Total {layer_type} Energy: {total_layer_energy:.2f}J ({layer_type_pct:.1f}% of model energy)",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if output_path:
            plt.savefig(output_path)
            print(f"Saved layer energy visualization to {output_path}")
        else:
            plt.show()
            
        return fig