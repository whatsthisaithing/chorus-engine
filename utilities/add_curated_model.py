#!/usr/bin/env python3
"""
Utility for adding pre-tested models to curated_models.json.

Usage:
    python utilities/add_curated_model.py

Interactive prompts will guide you through adding a new model.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import list_repo_files, HfApi


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default."""
    if default:
        prompt = f"{prompt} [{default}]"
    prompt += ": "
    
    value = input(prompt).strip()
    return value if value else (default or "")


def get_bool(prompt: str, default: bool = False) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    value = input(f"{prompt} ({default_str}): ").strip().lower()
    
    if not value:
        return default
    
    return value in ("y", "yes", "true", "1")


def get_float(prompt: str, default: Optional[float] = None) -> float:
    """Get float input."""
    while True:
        value = get_input(prompt, str(default) if default else None)
        try:
            return float(value)
        except ValueError:
            print("Invalid number, try again")


def get_int(prompt: str, default: Optional[int] = None) -> int:
    """Get integer input."""
    while True:
        value = get_input(prompt, str(default) if default else None)
        try:
            return int(value)
        except ValueError:
            print("Invalid integer, try again")


def get_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """Get choice from list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")
    
    while True:
        value = input(f"Select 1-{len(choices)} [{choices.index(default) + 1 if default else ''}]: ").strip()
        
        if not value and default:
            return default
        
        try:
            idx = int(value) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        
        print("Invalid selection, try again")


def get_rating(prompt: str) -> str:
    """Get performance rating."""
    choices = ["excellent", "very_good", "good", "fair", "poor"]
    return get_choice(prompt, choices, default="good")


def validate_hf_repo(repo_id: str) -> bool:
    """Check if HuggingFace repo exists and is accessible."""
    try:
        api = HfApi()
        api.repo_info(repo_id)
        return True
    except Exception as e:
        print(f"Error: Cannot access repo '{repo_id}': {e}")
        return False


def list_gguf_files(repo_id: str) -> List[str]:
    """List all GGUF files in HuggingFace repo."""
    try:
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        return sorted(gguf_files)
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def main():
    """Main entry point."""
    print("=" * 70)
    print("Add Model to Curated Library")
    print("=" * 70)
    print()
    
    # Path to curated models JSON
    json_path = Path(__file__).parent.parent / "chorus_engine" / "data" / "curated_models.json"
    
    if not json_path.exists():
        print(f"Error: Curated models file not found: {json_path}")
        return 1
    
    # Load existing models
    with open(json_path) as f:
        data = json.load(f)
    
    existing_ids = {m["id"] for m in data["models"]}
    
    print(f"Currently {len(data['models'])} curated models")
    print()
    
    # Basic info
    print("Basic Information")
    print("-" * 70)
    
    model_id = get_input("Model ID (e.g., 'qwen2.5-14b-instruct')")
    if model_id in existing_ids:
        print(f"Error: Model ID '{model_id}' already exists")
        return 1
    
    name = get_input("Display Name (e.g., 'Qwen 2.5 14B Instruct')")
    description = get_input("Description (one sentence)")
    
    print()
    print("HuggingFace Repository")
    print("-" * 70)
    
    # Validate repo
    while True:
        repo_id = get_input("HuggingFace Repo ID (e.g., 'Qwen/Qwen2.5-14B-Instruct-GGUF')")
        if validate_hf_repo(repo_id):
            break
    
    # List GGUF files
    print(f"\nScanning {repo_id} for GGUF files...")
    gguf_files = list_gguf_files(repo_id)
    
    if not gguf_files:
        print("Warning: No GGUF files found in repo")
        filename_template = get_input("Filename Template (e.g., 'model-{quant}.gguf')")
    else:
        print(f"Found {len(gguf_files)} GGUF files:")
        for f in gguf_files[:5]:
            print(f"  - {f}")
        if len(gguf_files) > 5:
            print(f"  ... and {len(gguf_files) - 5} more")
        
        # Try to detect template
        if gguf_files:
            sample = gguf_files[0]
            # Replace common quant patterns
            template = sample
            for quant in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32"]:
                template = template.replace(quant.lower(), "{quant}").replace(quant, "{quant}")
            
            filename_template = get_input("Filename Template", default=template)
        else:
            filename_template = get_input("Filename Template")
    
    print()
    print("Model Specifications")
    print("-" * 70)
    
    parameters = get_float("Parameters (in billions)")
    context_window = get_int("Context Window (tokens)", default=8192)
    
    category = get_choice(
        "Category",
        ["balanced", "creative", "technical", "advanced"],
        default="balanced"
    )
    
    print()
    tags_input = get_input("Tags (comma-separated, e.g., 'conversation,technical,tested')")
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
    
    print()
    tested = get_bool("Has this model been tested with Chorus Engine?", default=False)
    default_model = get_bool("Should this be the default model?", default=False)
    
    # Performance ratings
    print()
    print("Performance Ratings")
    print("-" * 70)
    
    performance = {
        "conversation": get_rating("Conversation quality"),
        "memory_extraction": get_rating("Memory extraction"),
        "prompt_following": get_rating("Prompt following"),
        "creativity": get_rating("Creativity"),
    }
    
    # Recommended quantizations
    print()
    print("Recommended Quantizations by VRAM")
    print("-" * 70)
    
    recommended_quant = {}
    vram_tiers = ["6GB", "8GB", "12GB", "16GB", "24GB", "32GB", "48GB"]
    
    print("Enter recommended quantization for each VRAM tier (or leave blank)")
    for tier in vram_tiers:
        quant = get_input(f"  {tier}")
        if quant:
            recommended_quant[tier] = quant
    
    # Quantizations
    print()
    print("Quantization Details")
    print("-" * 70)
    
    quantizations = []
    
    while True:
        print(f"\nQuantization #{len(quantizations) + 1}")
        
        quant = get_input("Quantization (e.g., 'Q4_K_M', or leave blank to finish)")
        if not quant:
            break
        
        filename = filename_template.replace("{quant}", quant)
        filename = get_input("Filename", default=filename)
        
        file_size_mb = get_int("File Size (MB)")
        min_vram_mb = get_int("Minimum VRAM (MB)")
        
        quantizations.append({
            "quant": quant,
            "filename": filename,
            "file_size_mb": file_size_mb,
            "min_vram_mb": min_vram_mb
        })
    
    if not quantizations:
        print("Error: At least one quantization required")
        return 1
    
    # Optional warning
    warning = get_input("Warning message (optional, e.g., 'Requires 48GB+ VRAM')")
    
    # Build model entry
    model_entry = {
        "id": model_id,
        "name": name,
        "description": description,
        "repo_id": repo_id,
        "filename_template": filename_template,
        "parameters": parameters,
        "context_window": context_window,
        "category": category,
        "tags": tags,
        "recommended_quant": recommended_quant,
        "tested": tested,
        "default": default_model,
        "performance": performance,
        "quantizations": quantizations
    }
    
    if warning:
        model_entry["warning"] = warning
    
    # Preview
    print()
    print("=" * 70)
    print("Model Entry Preview")
    print("=" * 70)
    print(json.dumps(model_entry, indent=2))
    print()
    
    # Confirm
    if not get_bool("Add this model to curated_models.json?", default=True):
        print("Cancelled")
        return 0
    
    # Add to models list
    data["models"].append(model_entry)
    
    # Update timestamp
    from datetime import datetime
    data["updated"] = datetime.now().strftime("%Y-%m-%d")
    
    # Save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print()
    print(f"✓ Successfully added '{name}' to curated models")
    print(f"  Total models: {len(data['models'])}")
    print()
    print("Next steps:")
    print("  1. Test the model with Chorus Engine")
    print("  2. Update performance ratings if needed")
    print("  3. Commit changes to git")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
