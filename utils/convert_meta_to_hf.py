#!/usr/bin/env python3
"""Convert Meta Llama format to Hugging Face format."""

import argparse
import torch
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

def convert_meta_to_hf(meta_path, output_path):
    """Convert Meta's Llama format to Hugging Face format.
    
    Args:
        meta_path: Path to Meta format model
        output_path: Path to save HF format model
    """
    meta_path = Path(meta_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load params.json for config
    params_path = meta_path / "params.json"
    if not params_path.exists():
        raise ValueError(f"params.json not found in {meta_path}")
    
    # Create config
    config = LlamaConfig.from_pretrained(str(meta_path))
    config.save_pretrained(str(output_path))
    
    # Load weights
    consolidate_path = meta_path / "consolidated.00.pth"
    if not consolidate_path.exists():
        raise ValueError(f"consolidated.00.pth not found in {meta_path}")
    
    state_dict = torch.load(consolidate_path, map_location="cpu")
    
    # Create model with config
    model = LlamaForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    
    # Save model
    model.save_pretrained(str(output_path))
    
    # Copy tokenizer if it exists
    tokenizer_path = meta_path / "tokenizer.model"
    if tokenizer_path.exists():
        tokenizer = LlamaTokenizer.from_pretrained(str(meta_path))
        tokenizer.save_pretrained(str(output_path))
    
    print(f"Converted model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Meta Llama format to Hugging Face format")
    parser.add_argument("--meta_path", type=str, required=True, help="Path to Meta format model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save HF format model")
    args = parser.parse_args()
    
    convert_meta_to_hf(args.meta_path, args.output_path)
