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
    
    def visualize_all_layer_types(self, output_path=None):
        """Generate a summary visualization of energy consumption across all layer types.
        
        Args:
            output_path: Path to save the visualization
        """
        if not self.layer_measurements:
            raise ValueError("No measurements available. Run profile_all_layers first.")
        
        # Create a new figure for the summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Data to collect
        layer_types = []
        total_energy_contributions = []
        total_model_energy = 0
        
        # Process each layer type
        for layer_type, results in self.layer_measurements.items():
            # Get baseline energy for this layer type
            baseline_energy = results["baseline"]["avg_energy"]
            if total_model_energy == 0:  # Only set this once
                total_model_energy = baseline_energy
            
            # Calculate total energy contribution for this layer type
            layer_energy = 0
            for key, data in results.items():
                if key != "baseline" and "layer_energy_contribution" in data:
                    layer_energy += data["layer_energy_contribution"]
            
            layer_types.append(layer_type)
            total_energy_contributions.append(layer_energy)
        
        # Sort by energy contribution (descending)
        sorted_data = sorted(zip(layer_types, total_energy_contributions), 
                             key=lambda x: x[1], reverse=True)
        layer_types = [x[0] for x in sorted_data]
        total_energy_contributions = [x[1] for x in sorted_data]
        
        # Calculate percentages
        energy_percentages = [100 * e / total_model_energy for e in total_energy_contributions]
        
        # Bar chart with absolute energy contributions
        bars = ax1.bar(layer_types, total_energy_contributions)
        
        # Add values on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{total_energy_contributions[i]:.2f} J',
                    ha='center', va='bottom', rotation=0)
        
        # Add labels and title
        ax1.set_xlabel('Layer Type')
        ax1.set_ylabel('Energy Contribution (Joules)')
        ax1.set_title('Energy Contribution by Layer Type')
        
        # Add grid
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Stacked bar showing percentage breakdown of model energy
        bottom = 0
        for i, (layer_type, percentage) in enumerate(zip(layer_types, energy_percentages)):
            ax2.bar([0], [percentage], bottom=[bottom], label=f'{layer_type} ({percentage:.1f}%)')
            if percentage > 3:  # Only add text if segment is large enough
                ax2.text(0, bottom + percentage/2, f'{percentage:.1f}%', 
                        ha='center', va='center', color='white', fontweight='bold')
            bottom += percentage
        
        # Calculate other energy if total < 100%
        remaining = 100 - sum(energy_percentages)
        if remaining > 0.1:  # Only show if significant
            ax2.bar([0], [remaining], bottom=[bottom], label=f'Other ({remaining:.1f}%)')
            if remaining > 3:
                ax2.text(0, bottom + remaining/2, f'{remaining:.1f}%', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        # Set labels and title
        ax2.set_title('Energy Distribution by Layer Type')
        ax2.set_ylabel('Percentage of Total Energy')
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Model Components'])
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Set the y-axis limit to 100%
        ax2.set_ylim(0, 100)
        
        # Add summary text
        profiled_energy = sum(total_energy_contributions)
        profiled_pct = 100 * profiled_energy / total_model_energy
        plt.figtext(0.5, 0.01, 
                   f"Total Profiled Energy: {profiled_energy:.2f}J ({profiled_pct:.1f}% of model energy)",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if output_path:
            plt.savefig(output_path)
            print(f"Saved layer type energy visualization to {output_path}")
        else:
            plt.show()
            
        return fig

    def visualize_all_layers(self, output_path=None):
        """Generate visualization showing energy contribution of each individual layer.
        
        Args:
            output_path: Path to save the visualization
        """
        if not self.layer_measurements:
            raise ValueError("No measurements available. Run profile_all_layers first.")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data to collect
        all_layer_names = []
        all_energy_contributions = []
        layer_types = []
        
        # Total model energy (use the first baseline found)
        total_model_energy = next(iter(self.layer_measurements.values()))["baseline"]["avg_energy"]
        
        # Process each layer type
        for layer_type, results in self.layer_measurements.items():
            layer_types.append(layer_type)
            
            # Extract layer data
            for key, data in results.items():
                if key != "baseline" and "layer_energy_contribution" in data:
                    # Extract position from key (e.g., LlamaMLP_5 -> 5)
                    position = int(key.split('_')[-1])
                    layer_name = f"{layer_type}_{position}"
                    
                    all_layer_names.append(layer_name)
                    all_energy_contributions.append(data["layer_energy_contribution"])
        
        # Sort by energy contribution (descending)
        sorted_data = sorted(zip(all_layer_names, all_energy_contributions), 
                             key=lambda x: x[1], reverse=True)
        all_layer_names = [x[0] for x in sorted_data]
        all_energy_contributions = [x[1] for x in sorted_data]
        
        # Create color mapping for layer types
        unique_types = list(set(name.split('_')[0] for name in all_layer_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        color_map = {layer_type: colors[i] for i, layer_type in enumerate(unique_types)}
        
        # Assign colors based on layer type
        bar_colors = [color_map[name.split('_')[0]] for name in all_layer_names]
        
        # Bar chart with absolute energy contributions
        bars = ax.bar(all_layer_names, all_energy_contributions, color=bar_colors)
        
        # Add values on top of bars (only for significant contributors)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > max(all_energy_contributions) * 0.05:  # Only label significant bars
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f} J',
                        ha='center', va='bottom', rotation=90, fontsize=8)
        
        # Add labels and title
        ax.set_xlabel('Layer')
        ax.set_ylabel('Energy Contribution (Joules)')
        ax.set_title('Energy Contribution by Individual Layer')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90, fontsize=8)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Create legend for layer types
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=color_map[t], label=t) 
                          for t in unique_types]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add summary text
        total_profiled = sum(all_energy_contributions)
        profiled_pct = 100 * total_profiled / total_model_energy
        plt.figtext(0.5, 0.01, 
                   f"Total Profiled Energy: {total_profiled:.2f}J ({profiled_pct:.1f}% of model energy)",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if output_path:
            plt.savefig(output_path)
            print(f"Saved individual layer energy visualization to {output_path}")
        else:
            plt.show()
            
        return fig