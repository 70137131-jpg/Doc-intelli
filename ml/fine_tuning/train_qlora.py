"""
QLoRA Fine-Tuning Pipeline for Document Intelligence.

Fine-tunes Mistral 7B (or Llama 3 8B) with QLoRA for document tasks:
summarization, extraction, classification, and question answering.

Requires: GPU with 16GB+ VRAM (A100 40GB recommended).
Run on: Google Colab Pro, RunPod, Lambda Labs, or local GPU.

Usage:
    python train_qlora.py --config config.yaml
    python train_qlora.py --config config.yaml --wandb_key YOUR_KEY
    python train_qlora.py --sweep                    # Run multi-experiment sweep
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(filepath: str) -> list[dict]:
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_sample(sample: dict) -> str:
    """Format a sample using the Alpaca template."""
    if sample.get("input", "").strip():
        return ALPACA_TEMPLATE.format(
            instruction=sample["instruction"],
            input=sample["input"],
            output=sample["output"],
        )
    return ALPACA_TEMPLATE_NO_INPUT.format(
        instruction=sample["instruction"],
        output=sample["output"],
    )


def tokenize_dataset(
    samples: list[dict],
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    """Tokenize the dataset for training."""
    formatted = [format_sample(s) for s in samples]

    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    # Set labels = input_ids for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()

    return Dataset.from_dict(tokenized)


SWEEP_GRID = {
    "learning_rate": [1e-4, 5e-5, 2e-5],
    "lora_r": [8, 16, 32],
    "lora_alpha": [16, 32],
    "num_epochs": [3, 5],
}


def run_sweep(base_config_path: str, wandb_key: str | None = None):
    """Run multi-experiment parameter sweep and compare results.

    Iterates through learning rate x LoRA rank combinations, trains each,
    and produces a comparison summary.
    """
    import copy
    import itertools

    base_config = load_config(base_config_path)
    results = []

    lr_values = SWEEP_GRID["learning_rate"]
    rank_values = SWEEP_GRID["lora_r"]
    combos = list(itertools.product(lr_values, rank_values))

    print(f"Multi-experiment sweep: {len(combos)} configurations")
    print(f"  Learning rates: {lr_values}")
    print(f"  LoRA ranks: {rank_values}")
    print("=" * 60)

    for idx, (lr, rank) in enumerate(combos):
        exp_name = f"exp_{idx:02d}_lr{lr}_r{rank}"
        print(f"\n{'='*60}")
        print(f"Experiment {idx+1}/{len(combos)}: {exp_name}")
        print(f"  lr={lr}, lora_r={rank}")
        print("=" * 60)

        config = copy.deepcopy(base_config)
        config["training"]["learning_rate"] = lr
        config["qlora"]["r"] = rank
        config["output"]["model_dir"] = os.path.join(base_config["output"]["model_dir"], exp_name)
        config["output"]["logs_dir"] = os.path.join(base_config["output"]["logs_dir"], exp_name)

        if wandb_key:
            config.setdefault("wandb", {})
            config["wandb"]["run_name"] = exp_name

        os.makedirs(config["output"]["model_dir"], exist_ok=True)
        os.makedirs(config["output"]["logs_dir"], exist_ok=True)

        # Write temp config
        temp_config_path = os.path.join(config["output"]["model_dir"], "config.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        try:
            eval_results = run_single_experiment(config, wandb_key)
            eval_results["experiment"] = exp_name
            eval_results["learning_rate"] = lr
            eval_results["lora_r"] = rank
            results.append(eval_results)
            print(f"  Result: eval_loss={eval_results.get('eval_loss', 'N/A')}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"experiment": exp_name, "error": str(e), "learning_rate": lr, "lora_r": rank})

    # Summary report
    print(f"\n{'='*60}")
    print("SWEEP RESULTS SUMMARY")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x.get("eval_loss", float("inf"))):
        loss = r.get("eval_loss", "ERROR")
        print(f"  {r['experiment']:30s}  eval_loss={loss}")

    # Save results
    summary_path = os.path.join(base_config["output"]["model_dir"], "sweep_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results saved to {summary_path}")
    return results


def run_single_experiment(config: dict, wandb_key: str | None = None) -> dict:
    """Run a single training experiment from a config dict. Returns eval results."""
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    wandb_config = config.get("wandb", {})
    report_to = "wandb" if wandb_config.get("project") else "none"

    quant_config = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
    )

    model_name = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    qlora_config = config["qlora"]
    lora_config = LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config.get("lora_alpha", qlora_config["r"] * 2),
        lora_dropout=qlora_config["lora_dropout"],
        target_modules=qlora_config["target_modules"],
        bias=qlora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    dataset_config = config["dataset"]
    train_data = load_jsonl(dataset_config["train_file"])
    val_data = load_jsonl(dataset_config["val_file"])
    if dataset_config.get("max_samples"):
        train_data = train_data[: dataset_config["max_samples"]]
        val_data = val_data[: min(len(val_data), dataset_config["max_samples"] // 5)]

    train_config = config["training"]
    train_dataset = tokenize_dataset(train_data, tokenizer, train_config["max_seq_length"])
    val_dataset = tokenize_dataset(val_data, tokenizer, train_config["max_seq_length"])

    output_config = config["output"]
    training_args = TrainingArguments(
        output_dir=output_config["model_dir"],
        num_train_epochs=train_config["num_epochs"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        weight_decay=train_config["weight_decay"],
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        optim=train_config.get("optim", "paged_adamw_8bit"),
        logging_steps=train_config["logging_steps"],
        logging_dir=output_config["logs_dir"],
        save_strategy=train_config["save_strategy"],
        evaluation_strategy=train_config["evaluation_strategy"],
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        report_to=report_to,
        run_name=wandb_config.get("run_name"),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_config["model_dir"])
    tokenizer.save_pretrained(output_config["model_dir"])

    eval_results = trainer.evaluate()
    with open(os.path.join(output_config["model_dir"], "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--sweep", action="store_true", help="Run multi-experiment parameter sweep")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.config, args.wandb_key)
        return

    config = load_config(args.config)

    # Setup W&B
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    wandb_config = config.get("wandb", {})
    if wandb_config.get("project"):
        os.environ["WANDB_PROJECT"] = wandb_config["project"]
        report_to = "wandb"
    else:
        report_to = "none"

    # Quantization config
    quant_config = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
    )

    # Load model
    model_name = config["base_model"]
    print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    qlora_config = config["qlora"]
    lora_config = LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config["lora_alpha"],
        lora_dropout=qlora_config["lora_dropout"],
        target_modules=qlora_config["target_modules"],
        bias=qlora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset_config = config["dataset"]
    print(f"Loading training data from: {dataset_config['train_file']}")

    train_data = load_jsonl(dataset_config["train_file"])
    val_data = load_jsonl(dataset_config["val_file"])

    if dataset_config.get("max_samples"):
        train_data = train_data[: dataset_config["max_samples"]]
        val_data = val_data[: min(len(val_data), dataset_config["max_samples"] // 5)]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Tokenize
    train_config = config["training"]
    max_seq_length = train_config["max_seq_length"]

    train_dataset = tokenize_dataset(train_data, tokenizer, max_seq_length)
    val_dataset = tokenize_dataset(val_data, tokenizer, max_seq_length)

    # Training arguments
    output_config = config["output"]
    training_args = TrainingArguments(
        output_dir=output_config["model_dir"],
        num_train_epochs=train_config["num_epochs"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        warmup_ratio=train_config["warmup_ratio"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        weight_decay=train_config["weight_decay"],
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        optim=train_config.get("optim", "paged_adamw_8bit"),
        logging_steps=train_config["logging_steps"],
        logging_dir=output_config["logs_dir"],
        save_strategy=train_config["save_strategy"],
        evaluation_strategy=train_config["evaluation_strategy"],
        load_best_model_at_end=train_config["load_best_model_at_end"],
        metric_for_best_model=train_config["metric_for_best_model"],
        report_to=report_to,
        run_name=wandb_config.get("run_name"),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting QLoRA fine-tuning...")
    print(f"  Base model: {model_name}")
    print(f"  LoRA rank: {qlora_config['r']}")
    print(f"  Learning rate: {train_config['learning_rate']}")
    print(f"  Epochs: {train_config['num_epochs']}")
    print(f"  Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    print("=" * 60 + "\n")

    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {output_config['model_dir']}")
    model.save_pretrained(output_config["model_dir"])
    tokenizer.save_pretrained(output_config["model_dir"])

    # Evaluate
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Eval results: {eval_results}")

    with open(os.path.join(output_config["model_dir"], "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
