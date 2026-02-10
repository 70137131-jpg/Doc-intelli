"""
Load and prepare the FUNSD dataset for LayoutLMv3 token classification training.

FUNSD (Form Understanding in Noisy Scanned Documents) contains 199 annotated forms
with word-level bounding boxes and semantic entity labels:
  - question (form field labels)
  - answer (form field values)
  - header (section headers)
  - other (non-entity text)

Usage:
    python load_funsd.py --output ./data/funsd_processed
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict, load_dataset

# FUNSD NER label mapping
FUNSD_LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
LABEL2ID = {label: i for i, label in enumerate(FUNSD_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(FUNSD_LABELS)}


def normalize_bbox(bbox, width, height):
    """Normalize bounding box coordinates to 0-1000 range (LayoutLMv3 convention)."""
    return [
        int(1000 * bbox[0] / width) if width > 0 else 0,
        int(1000 * bbox[1] / height) if height > 0 else 0,
        int(1000 * bbox[2] / width) if width > 0 else 0,
        int(1000 * bbox[3] / height) if height > 0 else 0,
    ]


def process_funsd_example(example):
    """Process a single FUNSD example into LayoutLMv3-compatible format.

    FUNSD provides word-level annotations with bounding boxes and NER tags.
    We convert to BIO format for token classification.
    """
    words = example.get("words", example.get("tokens", []))
    bboxes = example.get("bboxes", [])
    ner_tags = example.get("ner_tags", [])
    image = example.get("image", None)

    # Get image dimensions for normalization
    if image is not None:
        width, height = image.size
    else:
        width, height = 1000, 1000

    normalized_bboxes = []
    for bbox in bboxes:
        normalized_bboxes.append(normalize_bbox(bbox, width, height))

    # Map integer NER tags to label strings
    labels = []
    for tag in ner_tags:
        if isinstance(tag, int):
            labels.append(FUNSD_LABELS[tag] if tag < len(FUNSD_LABELS) else "O")
        else:
            labels.append(tag if tag in LABEL2ID else "O")

    return {
        "words": words,
        "bboxes": normalized_bboxes,
        "ner_tags": [LABEL2ID.get(l, 0) for l in labels],
        "ner_labels": labels,
        "image_width": width,
        "image_height": height,
    }


def load_funsd(output_dir: str = None) -> DatasetDict:
    """Load FUNSD from HuggingFace and process for LayoutLMv3 training.

    Returns a DatasetDict with train/test splits containing:
      - words: list of word strings
      - bboxes: list of normalized [x0, y0, x1, y1] bounding boxes
      - ner_tags: list of integer NER labels (BIO format)
      - ner_labels: list of string NER labels
    """
    print("Loading FUNSD dataset from HuggingFace...")
    ds = load_dataset("nielsr/funsd")

    processed_splits = {}
    for split_name in ds:
        print(f"Processing {split_name} split ({len(ds[split_name])} examples)...")
        processed = []
        for example in ds[split_name]:
            proc = process_funsd_example(example)
            if proc["words"]:
                processed.append(proc)

        processed_splits[split_name] = Dataset.from_dict({
            "words": [p["words"] for p in processed],
            "bboxes": [p["bboxes"] for p in processed],
            "ner_tags": [p["ner_tags"] for p in processed],
            "ner_labels": [p["ner_labels"] for p in processed],
        })
        print(f"  {split_name}: {len(processed)} examples")

    dataset = DatasetDict(processed_splits)

    # Print label distribution
    all_tags = []
    for split in dataset:
        for tags in dataset[split]["ner_labels"]:
            all_tags.extend(tags)
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"\nLabel distribution across all splits:")
    for label, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        print(f"\nSaved to {output_dir}")

        # Save metadata
        metadata = {
            "dataset": "FUNSD",
            "labels": FUNSD_LABELS,
            "label2id": LABEL2ID,
            "num_labels": len(FUNSD_LABELS),
            "splits": {s: len(dataset[s]) for s in dataset},
            "label_distribution": tag_counts,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Load and process FUNSD dataset")
    parser.add_argument("--output", type=str, default="./data/funsd_processed")
    args = parser.parse_args()

    load_funsd(args.output)


if __name__ == "__main__":
    main()
