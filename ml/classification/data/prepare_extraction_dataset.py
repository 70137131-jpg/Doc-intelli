"""
Build a unified extraction training dataset from FUNSD + CORD.

Combines both datasets into a single format suitable for training
LayoutLMv3 (token classification) or instruction-tuned extraction models.

Usage:
    python prepare_extraction_dataset.py --output ./data/extraction_dataset
    python prepare_extraction_dataset.py --output ./data/extraction_dataset --funsd_only
    python prepare_extraction_dataset.py --output ./data/extraction_dataset --cord_only
"""

import argparse
import json
import os
import random

from datasets import Dataset, DatasetDict

from load_funsd import load_funsd
from load_cord import load_cord


def funsd_to_extraction_format(funsd_dataset: DatasetDict) -> list[dict]:
    """Convert FUNSD data to a unified extraction format.

    Extracts question-answer pairs from form annotations as field extractions.
    """
    samples = []

    for split_name in funsd_dataset:
        for example in funsd_dataset[split_name]:
            words = example["words"]
            labels = example["ner_labels"]

            # Extract contiguous entity spans
            entities = []
            current_entity = None
            current_words = []

            for word, label in zip(words, labels):
                if label.startswith("B-"):
                    if current_entity:
                        entities.append({
                            "field_name": current_entity.lower(),
                            "field_value": " ".join(current_words),
                        })
                    current_entity = label[2:]
                    current_words = [word]
                elif label.startswith("I-") and current_entity:
                    current_words.append(word)
                else:
                    if current_entity:
                        entities.append({
                            "field_name": current_entity.lower(),
                            "field_value": " ".join(current_words),
                        })
                        current_entity = None
                        current_words = []

            # Final entity
            if current_entity:
                entities.append({
                    "field_name": current_entity.lower(),
                    "field_value": " ".join(current_words),
                })

            if entities:
                # Build full text from all words
                full_text = " ".join(words)
                samples.append({
                    "text": full_text,
                    "fields": entities,
                    "field_count": len(entities),
                    "source": "funsd",
                    "doc_type": "form",
                    "split": split_name,
                })

    return samples


def cord_to_extraction_format(cord_dataset: DatasetDict) -> list[dict]:
    """Convert CORD data to a unified extraction format."""
    samples = []

    for split_name in cord_dataset:
        for example in cord_dataset[split_name]:
            fields = json.loads(example["fields"])
            if fields:
                samples.append({
                    "text": example["text"],
                    "fields": [{"field_name": f["field_name"], "field_value": f["field_value"]}
                               for f in fields],
                    "field_count": len(fields),
                    "source": "cord",
                    "doc_type": "receipt",
                    "split": split_name,
                })

    return samples


def to_instruction_format(samples: list[dict]) -> list[dict]:
    """Convert extraction samples to instruction-tuning format (Alpaca style).

    This allows the same data to be used for QLoRA fine-tuning.
    """
    instruction_samples = []

    for sample in samples:
        doc_type = sample.get("doc_type", "document")

        # Build the expected JSON output
        output_fields = {}
        for f in sample["fields"]:
            name = f["field_name"]
            if name in output_fields:
                if isinstance(output_fields[name], list):
                    output_fields[name].append(f["field_value"])
                else:
                    output_fields[name] = [output_fields[name], f["field_value"]]
            else:
                output_fields[name] = f["field_value"]

        instruction_samples.append({
            "instruction": f"Extract all key fields from this {doc_type} as structured JSON.",
            "input": sample["text"],
            "output": json.dumps(output_fields, indent=2),
            "task_type": "extraction",
            "doc_type": doc_type,
            "source": sample["source"],
        })

    return instruction_samples


def build_unified_dataset(
    output_dir: str,
    include_funsd: bool = True,
    include_cord: bool = True,
    include_instruction_format: bool = True,
):
    """Build the unified extraction dataset from FUNSD and CORD."""
    all_samples = []

    if include_funsd:
        print("=" * 60)
        print("Loading FUNSD...")
        funsd_ds = load_funsd()
        funsd_samples = funsd_to_extraction_format(funsd_ds)
        all_samples.extend(funsd_samples)
        print(f"  FUNSD contributed {len(funsd_samples)} samples")

    if include_cord:
        print("=" * 60)
        print("Loading CORD...")
        cord_ds = load_cord()
        cord_samples = cord_to_extraction_format(cord_ds)
        all_samples.extend(cord_samples)
        print(f"  CORD contributed {len(cord_samples)} samples")

    print(f"\n{'=' * 60}")
    print(f"Total extraction samples: {len(all_samples)}")

    # Split into train/val/test (respect original splits where possible)
    train_samples = [s for s in all_samples if s.get("split") in ("train", "training")]
    test_samples = [s for s in all_samples if s.get("split") in ("test", "testing")]
    val_samples = [s for s in all_samples if s.get("split") in ("validation", "val")]

    # If no validation split, carve from train
    if not val_samples and train_samples:
        random.shuffle(train_samples)
        val_size = max(1, int(len(train_samples) * 0.1))
        val_samples = train_samples[:val_size]
        train_samples = train_samples[val_size:]

    # Samples without a recognized split go to train
    unassigned = [s for s in all_samples if s.get("split") not in
                  ("train", "training", "test", "testing", "validation", "val")]
    if unassigned:
        random.shuffle(unassigned)
        n = len(unassigned)
        train_samples.extend(unassigned[:int(n * 0.8)])
        val_samples.extend(unassigned[int(n * 0.8):int(n * 0.9)])
        test_samples.extend(unassigned[int(n * 0.9):])

    def to_dataset(data):
        return Dataset.from_dict({
            "text": [s["text"] for s in data],
            "fields": [json.dumps(s["fields"]) for s in data],
            "field_count": [s["field_count"] for s in data],
            "source": [s["source"] for s in data],
            "doc_type": [s["doc_type"] for s in data],
        })

    dataset = DatasetDict({
        "train": to_dataset(train_samples),
        "validation": to_dataset(val_samples),
        "test": to_dataset(test_samples),
    })

    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    print(f"\nSaved unified extraction dataset to {output_dir}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")

    # Also save instruction-tuning format as JSONL
    if include_instruction_format:
        instruction_dir = os.path.join(output_dir, "instruction_format")
        os.makedirs(instruction_dir, exist_ok=True)

        for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            instr_data = to_instruction_format(split_data)
            filepath = os.path.join(instruction_dir, f"{split_name}.jsonl")
            with open(filepath, "w") as f:
                for item in instr_data:
                    f.write(json.dumps(item) + "\n")
            print(f"  Instruction format {split_name}: {len(instr_data)} samples → {filepath}")

    # Save metadata
    source_counts = {}
    for s in all_samples:
        src = s["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    metadata = {
        "total_samples": len(all_samples),
        "splits": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
        },
        "source_distribution": source_counts,
        "sources_included": {
            "funsd": include_funsd,
            "cord": include_cord,
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build unified extraction dataset")
    parser.add_argument("--output", type=str, default="./data/extraction_dataset")
    parser.add_argument("--funsd_only", action="store_true", help="Only include FUNSD")
    parser.add_argument("--cord_only", action="store_true", help="Only include CORD")
    parser.add_argument("--no_instruction", action="store_true",
                        help="Skip generating instruction-tuning format")
    args = parser.parse_args()

    include_funsd = not args.cord_only
    include_cord = not args.funsd_only

    build_unified_dataset(
        output_dir=args.output,
        include_funsd=include_funsd,
        include_cord=include_cord,
        include_instruction_format=not args.no_instruction,
    )


if __name__ == "__main__":
    main()
