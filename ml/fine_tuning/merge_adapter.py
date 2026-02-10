"""
Merge LoRA adapter weights into the base model for deployment.

Usage:
    python merge_adapter.py --adapter_dir ./output/qlora-adapter --output_dir ./output/merged-model
    python merge_adapter.py --adapter_dir ./output/qlora-adapter --output_dir ./output/merged-model --push_to_hub username/model-name
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_save(adapter_dir: str, output_dir: str, push_to_hub: str | None = None):
    print(f"Loading adapter from: {adapter_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    # Determine base model from adapter config
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_name}")

    # Load base model in fp16
    print("Loading base model in fp16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and merge adapter
    print("Merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Create model card
    model_card = f"""---
base_model: {base_model_name}
tags:
  - document-intelligence
  - qlora
  - fine-tuned
license: apache-2.0
---

# Document Intelligence Fine-Tuned Model

Fine-tuned from `{base_model_name}` using QLoRA for document intelligence tasks.

## Tasks
- Document summarization
- Key field extraction (Invoice, Contract, Resume, etc.)
- Document classification with reasoning
- Document-grounded question answering

## Training Details
- Method: QLoRA (4-bit quantization + LoRA)
- LoRA rank: {peft_config.r}
- LoRA alpha: {peft_config.lora_alpha}
- Target modules: {', '.join(peft_config.target_modules)}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

prompt = \"\"\"### Instruction:
Summarize the following document in 2-3 sentences.

### Input:
[Your document text here]

### Response:
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)

    # Push to Hub if requested
    if push_to_hub:
        print(f"Pushing to Hugging Face Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)

    print("Merge complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter_dir", type=str, default="./output/qlora-adapter")
    parser.add_argument("--output_dir", type=str, default="./output/merged-model")
    parser.add_argument("--push_to_hub", type=str, default=None)
    args = parser.parse_args()

    merge_and_save(args.adapter_dir, args.output_dir, args.push_to_hub)
