import os
import threading
import time
from queue import Queue
import json
import numpy as np
import pynvml
import torch

class PowerMeasurement:
    """Container for a single power measurement with metadata."""
    def __init__(self, timestamp, power_mw, gpu_util, mem_util, temp, component=None, event=None):
        self.timestamp = timestamp
        self.power_mw = power_mw  # in milliwatts
        self.power_w = power_mw / 1000.0  # in watts
        self.gpu_util = gpu_util  # in percent
        self.mem_util = mem_util  # in percent
        self.temp = temp  # in celsius
        self.component = component  # component identifier (if any)
        self.event = event  # event identifier (if any)
    
    def to_dict(self):
        """Convert measurement to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "power_mw": self.power_mw,
            "power_w": self.power_w,
            "gpu_util": self.gpu_util,
            "mem_util": self.mem_util,
            "temp": self.temp,
            "component": self.component,
            "event": self.event
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create measurement from dictionary."""
        return cls(
            data["timestamp"],
            data["power_mw"],
            data["gpu_util"],
            data["mem_util"],
            data["temp"],
            data.get("component"),
            data.get("event")
        )

class PowerProfiler:
    """Power profiling tool using NVML to measure GPU power consumption."""
    
    def __init__(self, device_id=0, sampling_rate_ms=10, baseline_samples=100):
        """Initialize the power profiler.
        
        Args:
            device_id: GPU device ID to profile (default: 0)
            sampling_rate_ms: Sampling rate in milliseconds (default: 50)
            baseline_samples: Number of samples to collect for baseline (default: 100)
        """
        self.device_id = device_id
        self.sampling_rate_ms = sampling_rate_ms
        self.sampling_interval_s = sampling_rate_ms / 1000.0
        self.baseline_samples = baseline_samples
        
        # Initialize NVML
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        if device_id >= self.device_count:
            raise ValueError(f"Device ID {device_id} out of range (0-{self.device_count-1})")
        
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.device_name = pynvml.nvmlDeviceGetName(self.handle)
        
        # For continuous sampling
        self.sampling_thread = None
        self.sampling_queue = Queue()
        self.stop_sampling = threading.Event()
        self.measurements = []
        
        # Measure baseline power
        self.baseline_power = self._measure_baseline()
        print(f"Baseline power for {self.device_name}: {self.baseline_power:.2f} W")
    
    def _measure_baseline(self):
        """Measure GPU idle power consumption to establish a baseline.
        
        Returns:
            Average baseline power in watts
        """
        print(f"Measuring baseline power over {self.baseline_samples} samples...")
        powers = []
        for _ in range(self.baseline_samples):
            powers.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)  # Convert to watts
            time.sleep(self.sampling_interval_s)
        
        return sum(powers) / len(powers)
    
    def _get_measurement(self, component=None, event=None):
        """Take a single power measurement.
        
        Args:
            component: Optional component identifier
            event: Optional event identifier
            
        Returns:
            PowerMeasurement object with current readings
        """
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # in milliwatts
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        gpu_util = util.gpu  # in percent
        mem_util = util.memory  # in percent
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)  # in celsius
        #print(f"Power: {power / 1000:.2f} W, GPU Util: {gpu_util}%, Mem Util: {mem_util}%, Temp: {temp}C") 
        return PowerMeasurement(
            time.time(),
            power,
            gpu_util,
            mem_util,
            temp,
            component,
            event
        )
    
    def _sampling_loop(self):
        """Background thread function for continuous power sampling."""
        while not self.stop_sampling.is_set():
            try:
                measurement = self._get_measurement()
                #print(f"Inserting Sampled power: {measurement.power_w:.2f} W, GPU Util: {measurement.gpu_util}%, Mem Util: {measurement.mem_util}%, Temp: {measurement.temp}C")
                self.sampling_queue.put(measurement)
                time.sleep(self.sampling_interval_s)
            except Exception as e:
                print(f"Error in sampling thread: {e}")
                break
    
    def start_continuous_sampling(self):
        """Start background thread for continuous power sampling."""
        if self.sampling_thread is not None and self.sampling_thread.is_alive():
            print("Sampling already in progress")
            return
        
        # Clear previous state
        self.stop_sampling.clear()
        self.measurements = []
        
        # Start sampling thread
        self.sampling_thread = threading.Thread(target=self._sampling_loop)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()
        print(f"Started continuous power sampling at {self.sampling_rate_ms}ms intervals")
    
    def stop_continuous_sampling(self):
        """Stop background sampling thread and process results.
        
        Returns:
            List of PowerMeasurement objects collected during sampling
        """
        if self.sampling_thread is None or not self.sampling_thread.is_alive():
            print("No sampling in progress")
            return self.measurements

        #wait until we get at least one more sample
        while self.sampling_queue.empty():
            time.sleep(0.1)

        # Stop sampling thread
        self.stop_sampling.set()
        self.sampling_thread.join(timeout=2.0)
        
        # Process all remaining measurements from queue
        while not self.sampling_queue.empty():
            self.measurements.append(self.sampling_queue.get())
        
        print(f"Stopped sampling. Collected {len(self.measurements)} measurements")
        return self.measurements
    
    def measure_operation(self, operation_fn, *args, component=None, **kwargs):
        """Measure power during a specific operation.
        
        Args:
            operation_fn: Function to execute and measure
            *args: Arguments to pass to operation_fn
            component: Component identifier for the measurement
            **kwargs: Keyword arguments to pass to operation_fn
            
        Returns:
            Tuple of (operation result, PowerMeasurements during operation)
        """
        self.start_continuous_sampling()
        # Wait for at least one measurement to be taken before starting operation
        initial_wait = 0
        max_wait = 1.0  # Maximum 1 second wait
        while not self.measurements and initial_wait < max_wait:
            time.sleep(0.05)
            initial_wait += 0.05

        start_time = time.time()
        print(f"Operation starting at: {start_time}") 

        # Execute operation
        try:
            result = operation_fn(*args, **kwargs)
        except Exception as e:
            self.stop_continuous_sampling()
            raise e
        
        end_time = time.time()
        print(f"Operation completed at: {end_time}, duration: {end_time - start_time:.3f}s")

        # Wait briefly to ensure we capture measurements after operation
        time.sleep(2 * self.sampling_interval_s)

        measurements = self.stop_continuous_sampling()
        # Debug info
        if measurements:
            print(f"First measurement time: {measurements[0].timestamp}, Last: {measurements[-1].timestamp}")
            print(f"Time range: {measurements[-1].timestamp - measurements[0].timestamp:.3f}s")
        
        # Use a much larger buffer to catch nearby measurements
        time_buffer = max(0.05, 5 * self.sampling_interval_s)  # at least 50ms buffer
 
        op_measurements = [m for m in measurements if 
                        (start_time - time_buffer) <= m.timestamp <= (end_time + time_buffer)]
        
        print(f"Total measurements: {len(measurements)}, filtered for operation: {len(op_measurements)}")
    
        # Add component info to measurements
        for m in op_measurements:
            m.component = component
        
        return result, op_measurements
    
    def calculate_energy(self, measurements):
        """Calculate energy consumption from a list of measurements.
        
        Energy is calculated as the integral of power over time.
        
        Args:
            measurements: List of PowerMeasurement objects
            
        Returns:
            Dictionary with total energy in joules and other statistics
        """
        if not measurements:
            return {
                "energy_joules": 0,
                "avg_power_watts": 0,
                "peak_power_watts": 0,
                "duration_seconds": 0,
                "baseline_power_watts": self.baseline_power,
                "energy_above_baseline_joules": 0
            }
        
        # Sort measurements by timestamp
        measurements = sorted(measurements, key=lambda m: m.timestamp)
        
        # Calculate time deltas between measurements
        timestamps = [m.timestamp for m in measurements]
        powers = [m.power_w for m in measurements]
        
        # Calculate energy using trapezoidal rule
        total_duration = timestamps[-1] - timestamps[0]
        
        # If only one measurement, estimate using sampling rate
        if len(measurements) == 1:
            energy_joules = powers[0] * self.sampling_interval_s
        else:
            # Use numpy for efficient calculation with trapezoidal rule
            energy_joules = np.trapz(powers, timestamps)
        
        # Calculate energy above baseline
        baseline_energy = self.baseline_power * total_duration
        energy_above_baseline = energy_joules - baseline_energy
        
        return {
            "energy_joules": energy_joules,
            "avg_power_watts": sum(powers) / len(powers),
            "peak_power_watts": max(powers),
            "duration_seconds": total_duration,
            "baseline_power_watts": self.baseline_power,
            "energy_above_baseline_joules": energy_above_baseline
        }
    
    def save_measurements(self, measurements, filename):
        """Save measurements to a JSON file.
        
        Args:
            measurements: List of PowerMeasurement objects
            filename: Path to output JSON file
        """
        data = {
            "device": {
                "id": self.device_id,
                "name": self.device_name
            },
            "baseline_power_watts": self.baseline_power,
            "sampling_rate_ms": self.sampling_rate_ms,
            "measurements": [m.to_dict() for m in measurements]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(measurements)} measurements to {filename}")
    
    def load_measurements(self, filename):
        """Load measurements from a JSON file.
        
        Args:
            filename: Path to input JSON file
            
        Returns:
            List of PowerMeasurement objects
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        measurements = [PowerMeasurement.from_dict(m) for m in data["measurements"]]
        print(f"Loaded {len(measurements)} measurements from {filename}")
        return measurements
    
    def __del__(self):
        """Clean up NVML on destruction."""
        try:
            if self.sampling_thread and self.sampling_thread.is_alive():
                self.stop_sampling.set()
                self.sampling_thread.join(timeout=1.0)
            pynvml.nvmlShutdown()
        except:
            pass