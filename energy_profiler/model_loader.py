"""Utility functions for loading LLama models in different formats."""

import os
import torch
from pathlib import Path

def load_llama_model(model_class, model_path, device="cuda:0"):
    """Load Llama model, handling both HF and Meta formats.
    
    Args:
        model_class: The model class from HF transformers
        model_path: Path to the model
        device: Device to load model on
        
    Returns:
        Loaded model on the specified device
    """
    model_path = Path(model_path)
    
    # Check if it's a Hugging Face format model
    hf_files = ["pytorch_model.bin", "model.safetensors", "config.json"]
    if any((model_path / f).exists() for f in hf_files):
        print(f"Loading model in Hugging Face format from {model_path}")
        model = model_class.from_pretrained(str(model_path))
        model.to(device)
        return model
    
    # Check if it's Meta's original format
    meta_files = ["consolidated.00.pth", "params.json", "tokenizer.model"]
    if all((model_path / f).exists() for f in meta_files):
        print(f"Loading model in Meta format from {model_path}")
        try:
            # First try the transformers way which works with some models
            model = model_class.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,  # Meta models typically need this
                low_cpu_mem_usage=True
            )
            model.to(device)
            return model
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Trying to load with llama.cpp format handler...")
            
            # Try to import the specialized Llama loader if available
            try:
                from transformers import LlamaForCausalLM, LlamaConfig
                
                # Load the configuration from params.json
                config = LlamaConfig.from_pretrained(str(model_path))
                
                # Initialize empty model
                model = LlamaForCausalLM(config)
                
                # Load weights from consolidated files
                consolidated_path = model_path / "consolidated.00.pth"
                state_dict = torch.load(consolidated_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                
                model.to(device)
                return model
            except Exception as e2:
                raise RuntimeError(f"Failed to load model in Meta format: {e2}")
    
    # Not a recognized format
    raise ValueError(
        f"Model at {model_path} is not in a recognized format. "
        "Please convert it to Hugging Face format first."
    )
