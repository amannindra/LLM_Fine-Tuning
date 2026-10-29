import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into its base causal language model."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("new_output/checkpoint-223"),
        help="Adapter checkpoint directory (default: new_output/checkpoint-223).",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the base model recorded in adapter_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("merged_adapters_new"),
        help="Directory to write the merged model and tokenizer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = args.adapter.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter config not found: {adapter_dir / 'adapter_config.json'}")
    if not (adapter_dir / "adapter_model.safetensors").is_file() and not (
        adapter_dir / "adapter_model.bin"
    ).is_file():
        raise FileNotFoundError(f"Adapter weights not found in {adapter_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose another --output-dir."
        )

    adapter_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model_name = args.base_model or adapter_config.base_model_name_or_path
    if not base_model_name:
        raise ValueError(
            "Base model is not recorded in adapter_config.json; pass --base-model."
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_dir}")
    print(f"Output: {output_dir}")
    print(f"Loading weights as {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = model.merge_and_unload(safe_merge=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
