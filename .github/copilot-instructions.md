# Copilot Instructions for Llama Energy Profiling

This document provides detailed instructions and guidelines for implementing the Llama energy profiling framework. These instructions serve as a reference throughout the development process.

## Core Development Principles

1. **Accuracy First**: Ensure measurements are as accurate as possible by:
   - Accounting for baseline power draw
   - Running multiple trials to establish statistical significance
   - Validating measurements against known benchmarks

2. **Minimal Overhead**: The measurement framework should:
   - Add minimal computational overhead to avoid skewing results
   - Use asynchronous collection where possible
   - Batch measurements to reduce I/O impact

3. **Component Isolation**: Create clear boundaries between:
   - Measurement collection
   - Model instrumentation
   - Data analysis and visualization

4. **Extensibility**: Design for:
   - Easy addition of new measurement points
   - Support for different Llama model versions
   - Pluggable visualization and analysis tools

## Technical Implementation Guidelines

### NVML Integration

1. Use `pynvml` Python library for accessing NVIDIA Management Library:
   ```python
   import pynvml
   pynvml.nvmlInit()
   # Get device handle for GPU 0
   handle = pynvml.nvmlDeviceGetHandleByIndex(0)
   # Get power reading in milliwatts
   power = pynvml.nvmlDeviceGetPowerUsage(handle)
   ```

2. Create a background sampling thread for continuous monitoring:
   - Sample at 10-100ms intervals depending on precision needs
   - Use thread-safe queues to collect measurements
   - Implement configurable sampling rates

3. Capture these metrics at minimum:
   - Power usage (watts)
   - GPU utilization (%)
   - Memory utilization (%)
   - Temperature (°C) for correlation

### Model Instrumentation

1. Use PyTorch hooks to instrument model components:
   ```python
   def hook_fn(module, input, output):
       # Take power measurement
       # Log module type, timestamp, and power
       pass
   
   # Attach to model layers
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Linear):
           module.register_forward_hook(hook_fn)
   ```

2. Create wrappers for key model components:
   - Attention mechanisms
   - Feed-forward networks
   - Embedding layers
   - Output layers

3. Implement timing instrumentation:
   - Use high-precision timers (e.g., `time.perf_counter_ns()`)
   - Correlate time with power measurements
   - Account for measurement overhead

### Data Collection Framework

1. Create a hierarchical data structure for measurements:
   - Model-level statistics
   - Layer-level measurements
   - Component-level detailed metrics

2. Implement these measurement types:
   - Point-in-time power readings
   - Time interval measurements (energy = average power × time)
   - Cumulative energy usage

3. Design a flexible logging system:
   - JSON-formatted logs for easy analysis
   - Support for streaming to file or database
   - Include rich metadata and context

### Analysis & Visualization

1. Implement these core visualizations:
   - Layer-wise energy comparison bar charts
   - Time-series power usage during inference
   - Energy breakdown by component type
   - Comparative analysis across model sizes

2. Create energy efficiency metrics:
   - Energy per token (joules/token)
   - Energy per layer per token
   - Relative efficiency between components

3. Generate insightful comparative views:
   - Layer-wise energy distribution
   - Attention vs. feed-forward energy usage
   - Energy scaling with input length

## Implementation Approach for Key Components

### 1. Power Measurement Core Module

```python
class PowerProfiler:
    def __init__(self, device_id=0, sampling_rate_ms=50):
        self.device_id = device_id
        self.sampling_rate = sampling_rate_ms
        self.baseline_power = self._measure_baseline()
        
    def _measure_baseline(self):
        """Measure GPU idle power consumption as baseline"""
        # Take multiple samples and average
        pass
        
    def start_continuous_sampling(self):
        """Start background thread for continuous power sampling"""
        pass
        
    def stop_continuous_sampling(self):
        """Stop background sampling thread and return results"""
        pass
        
    def measure_operation(self, operation_fn, *args, **kwargs):
        """Measure power during a specific operation"""
        # Start sampling
        # Execute operation
        # Stop sampling and process results
        pass
```

### 2. Model Layer Instrumentation

```python
class EnergyTracker:
    def __init__(self, model, power_profiler):
        self.model = model
        self.profiler = power_profiler
        self.measurements = {}
        self._attach_hooks()
        
    def _attach_hooks(self):
        """Attach measurement hooks to model layers"""
        for name, module in self.model.named_modules():
            # Attach appropriate hooks based on module type
            pass
            
    def layer_entry_hook(self, module, input):
        """Hook called before forward pass of a layer"""
        pass
        
    def layer_exit_hook(self, module, input, output):
        """Hook called after forward pass of a layer"""
        pass
        
    def get_energy_profile(self):
        """Return complete energy profile data"""
        pass
```

### 3. Measurement Orchestration

```python
class LlamaEnergyProfiler:
    def __init__(self, model_path, model_version="llama3_2", device="cuda:0"):
        self.model = self._load_model(model_path, model_version)
        self.power_profiler = PowerProfiler()
        self.energy_tracker = EnergyTracker(self.model, self.power_profiler)
        
    def _load_model(self, model_path, model_version):
        """Load the specified Llama model"""
        pass
        
    def profile_inference(self, input_text, detailed=True):
        """Run inference with energy profiling"""
        # Start profiling
        # Tokenize input
        # Run inference
        # Collect and process measurements
        pass
        
    def generate_report(self, output_path=None):
        """Generate comprehensive energy profile report"""
        pass
```

## Testing and Validation Strategies

1. **Unit Testing**:
   - Test power measurement accuracy against nvidia-smi
   - Validate hook attachment and execution
   - Test data collection and aggregation

2. **Integration Testing**:
   - Test with small Llama 3.2 model
   - Compare against baseline power measurements
   - Validate consistency across multiple runs

3. **Validation Methods**:
   - Cross-check energy numbers with global power measurements
   - Verify logical distribution (attention should use more than linear layers)
   - Compare total energy with nvidia-smi reported values

## Analysis Focuses

When analyzing the energy profile data, focus on:

1. **Component Impact**:
   - Which layers consume the most energy?
   - How does energy scale with layer depth?
   - What is the energy overhead of attention mechanisms?

2. **Scaling Patterns**:
   - How does energy consumption scale with model size?
   - What is the relationship between parameter count and energy?
   - How does energy usage scale with input length?

3. **Optimization Opportunities**:
   - Which components would benefit most from optimization?
   - Are there energy usage patterns that suggest inefficiencies?
   - How do different operations compare in energy efficiency?

## Deliverables Guidance

1. **Core Library**:
   - Clean, well-documented Python modules
   - Comprehensive test suite
   - Flexible configuration options

2. **Example Scripts**:
   - Basic profiling scripts for each Llama version
   - Comparative analysis examples
   - Visualization generation scripts

3. **Documentation**:
   - API documentation
   - Usage tutorials
   - Technical explanation of methods and limitations

4. **Visualization Assets**:
   - Interactive dashboards (if applicable)
   - Static plots and charts
   - Comparative visualizations

## Performance Considerations

1. **Measurement Overhead**:
   - Aim for <5% performance impact from measurement code
   - Implement sampling rate throttling if performance degrades

2. **GPU Memory Usage**:
   - Minimize additional memory footprint
   - Clean up measurement data periodically

3. **Analysis Efficiency**:
   - Process larger datasets in chunks
   - Implement efficient data structures for large measurement sets

## Final Notes

This implementation requires a careful balance between measurement granularity and accuracy. The focus should be on providing actionable insights into energy usage patterns rather than absolutely precise measurements at the finest granularity. Statistical approaches and multiple measurement runs will help establish confidence in the findings.

Remember that NVML provides board-level power measurements, so attribution to specific operations will require statistical inference rather than direct measurement. Design the system to acknowledge these limitations while still providing valuable insights into energy consumption patterns.