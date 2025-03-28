#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment to measure energy impact of feed-forward networks in Llama 3.2 model.

This script compares the energy consumption of a Llama 3.2 model with and without
feed-forward networks enabled, providing insights into their energy impact.
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from energy_profiler.power_profiler import PowerProfiler
from energy_profiler.model_profiler import LlamaEnergyProfiler

def create_results_directory():
    """Create a directory to store results if it doesn't exist."""
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def plot_energy_comparison(results, output_path):
    """Plot energy comparison between configurations.
    
    Args:
        results: Dictionary with energy statistics for each configuration
        output_path: Path to save the plot
    """
    configs = []
    energies = []
    errors = []
    
    for config, data in results.items():
        if "avg_energy" in data:
            configs.append(config)
            energies.append(data["avg_energy"])
            if "energy" in data and len(data["energy"]) > 1:
                errors.append(np.std(data["energy"]))
            else:
                errors.append(0)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the bar chart
    bars = ax.bar(configs, energies, yerr=errors, capsize=10)
    
    # Add values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{energies[i]:.2f} J',
                ha='center', va='bottom', rotation=0)
    
    # Calculate percentage differences
    if "baseline" in results and len(configs) > 1:
        baseline_energy = results["baseline"]["avg_energy"]
        for i, config in enumerate(configs):
            if config != "baseline":
                pct_diff = ((baseline_energy - energies[i]) / baseline_energy) * 100
                ax.text(i, energies[i] / 2,
                        f'{pct_diff:.1f}%\nsaved',
                        ha='center', va='center', color='white', fontweight='bold')
    
    # Add labels and title
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Energy Consumption (Joules)')
    ax.set_title('Energy Consumption Comparison')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved energy comparison plot to {output_path}")

def plot_power_over_time(measurements, output_path):
    """Plot power consumption over time.
    
    Args:
        measurements: List of PowerMeasurement objects
        output_path: Path to save the plot
    """
    # Extract timestamps and power values
    timestamps = []
    powers = []
    
    base_time = measurements[0].timestamp
    for m in measurements:
        timestamps.append(m.timestamp - base_time)  # Convert to relative time
        powers.append(m.power_w)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the line chart
    ax.plot(timestamps, powers, marker='.', markersize=3, linestyle='-', linewidth=1)
    
    # Add labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Power Consumption (Watts)')
    ax.set_title('Power Consumption Over Time')
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved power over time plot to {output_path}")

def run_experiment():
    """Run the energy profiling experiment."""
    parser = argparse.ArgumentParser(description="Energy profiling experiment for Llama 3.2")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Llama 3.2 model")
    parser.add_argument("--prompt", type=str, default="Explain how transformers work in artificial intelligence", 
                      help="Prompt for the model")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs for each configuration")
    
    args = parser.parse_args()
    
    try:
        # Import at runtime to avoid requiring these dependencies for importing the module
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformers import LlamaForCausalLM, LlamaTokenizer
        
        # Determine model class and components to profile
        # For Llama models specifically
        try:
            # Try to import Llama-specific FeedForward class
            from transformers.models.llama.modeling_llama import LlamaMLP as FeedForward
        except ImportError:
            # Fall back to a general approach
            print("Could not import Llama-specific FeedForward, using generic approach")
            # Look for Feed Forward class in the model
            FeedForward = None
            
        # Create results directory
        results_dir = create_results_directory()
        
        # Initialize tokenizer
        print(f"Loading tokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # Initialize energy profiler
        profiler = LlamaEnergyProfiler(AutoModelForCausalLM, args.model_path)
        
        # Perform a warm-up run to ensure everything is loaded
        print("Performing warm-up run...")
        profiler.profile_inference(args.prompt, tokenizer, max_tokens=10)
        
        # Find the feed-forward component class
        if FeedForward is None:
            # Look through the model to find the feed-forward class
            feed_forward_candidates = []
            for name, module in profiler.model.named_modules():
                if "mlp" in name.lower() or "feed_forward" in name.lower():
                    feed_forward_candidates.append((name, type(module)))
            
            if feed_forward_candidates:
                print(f"Found potential feed-forward components:")
                for name, cls in feed_forward_candidates:
                    print(f"  {name}: {cls.__name__}")
                
                # Use the first candidate
                FeedForward = feed_forward_candidates[0][1]
                print(f"Using {FeedForward.__name__} as the feed-forward component")
            else:
                raise ValueError("Could not identify feed-forward component in model")
        
        # Run the component comparison
        print(f"Running experiment with prompt: '{args.prompt}'")
        results = profiler.compare_components_energy(
            args.prompt, tokenizer, [FeedForward], 
            max_tokens=args.max_tokens, num_runs=args.num_runs
        )
        
        # Save the detailed results
        results_path = results_dir / "ffn_energy_results.json"
        with open(results_path, "w") as f:
            # Convert to serializable format
            serializable_results = {
                k: {
                    sk: sv for sk, sv in v.items() 
                    if not isinstance(sv, np.ndarray)  # Skip numpy arrays
                } 
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved detailed results to {results_path}")
        
        # Plot the energy comparison
        plot_path = results_dir / "ffn_energy_comparison.png"
        plot_energy_comparison(results, plot_path)
        
        # Run a single test with power measurements over time
        print("Running test for power over time plot...")
        profiler.reset_model()
        _, measurements = profiler.power_profiler.measure_operation(
            lambda: profiler.model.generate(
                inputs=tokenizer(args.prompt, return_tensors="pt").to(profiler.device)["input_ids"],
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                top_p=0.9
            )
        )
        
        # Plot power over time
        power_plot_path = results_dir / "power_over_time.png"
        plot_power_over_time(measurements, power_plot_path)
        
        # Print summary
        print("\n=== EXPERIMENT SUMMARY ===")
        print(f"Model: {args.model_path}")
        print(f"Prompt: '{args.prompt}'")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Number of runs: {args.num_runs}")
        print("\nEnergy consumption:")
        
        baseline_energy = results["baseline"]["avg_energy"]
        print(f"  Baseline: {baseline_energy:.2f} J")
        
        for config, data in results.items():
            if config != "baseline" and "avg_energy" in data:
                print(f"  {config}: {data['avg_energy']:.2f} J ({data['energy_saved_pct']:.1f}% saved)")
        
        print("\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_experiment())