"""
Fine-tune DeBERTa-v3-base for document classification.

Usage:
    python train_classifier.py --data_dir ./data/dataset --output_dir ./output/model
"""

import argparse
import os

import numpy as np
import yaml
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


CATEGORIES = ["Invoice", "Contract", "Report", "Resume", "Letter", "Other"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train document classifier")
    parser.add_argument("--data_dir", type=str, default="./data/dataset")
    parser.add_argument("--output_dir", type=str, default="./output/model")
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]
    max_length = config["model"]["max_length"]
    training_config = config["training"]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: cat for i, cat in enumerate(CATEGORIES)},
        label2id={cat: i for i, cat in enumerate(CATEGORIES)},
    )

    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = load_from_disk(args.data_dir)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "category"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        num_train_epochs=training_config["num_epochs"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        fp16=training_config.get("fp16", False),
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        logging_dir=config["output"]["logs_dir"],
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"])
    print(f"Test Results: {test_results}")

    # Save model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
