"""
Load and prepare the CORD dataset for receipt field extraction training.

CORD (Consolidated Receipt Dataset) contains 11,000 Indonesian receipt images
with structured annotations for 30+ field types including:
  - menu items (name, count, price, subtotal)
  - totals (total_price, tax, discount, service)
  - store info (name, address, telephone)

Usage:
    python load_cord.py --output ./data/cord_processed
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict, load_dataset

# CORD field categories — we group granular labels into broader field types
CORD_FIELD_GROUPS = {
    "menu": ["menu.nm", "menu.num", "menu.unitprice", "menu.cnt", "menu.discountprice",
             "menu.price", "menu.itemsubtotal", "menu.vatyn", "menu.etc", "menu.sub_nm",
             "menu.sub_unitprice", "menu.sub_cnt", "menu.sub_price", "menu.sub_etc"],
    "void_menu": ["void_menu.nm", "void_menu.price"],
    "subtotal": ["sub_total.subtotal_price", "sub_total.discount_price",
                 "sub_total.service_price", "sub_total.othersvc_price",
                 "sub_total.tax_price", "sub_total.etc"],
    "total": ["total.total_price", "total.total_etc", "total.cashprice",
              "total.changeprice", "total.creditcardprice", "total.emloybee_price",
              "total.menutype_cnt", "total.menuqty_cnt"],
}

# Simplified label set for our extraction model
CORD_LABELS = [
    "O",
    "B-MENU_ITEM", "I-MENU_ITEM",
    "B-MENU_PRICE", "I-MENU_PRICE",
    "B-SUBTOTAL", "I-SUBTOTAL",
    "B-TOTAL", "I-TOTAL",
    "B-TAX", "I-TAX",
    "B-STORE_NAME", "I-STORE_NAME",
    "B-STORE_ADDR", "I-STORE_ADDR",
    "B-DATE", "I-DATE",
    "B-OTHER", "I-OTHER",
]
LABEL2ID = {label: i for i, label in enumerate(CORD_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(CORD_LABELS)}


def map_cord_label(cord_label: str) -> str:
    """Map a CORD fine-grained label to our simplified label set."""
    if not cord_label or cord_label == "O":
        return "O"

    # Extract prefix (B- or I-) and field name
    prefix = ""
    field = cord_label
    if cord_label.startswith("B-") or cord_label.startswith("I-"):
        prefix = cord_label[:2]
        field = cord_label[2:]

    if field in ("menu.nm", "menu.sub_nm"):
        return f"{prefix}MENU_ITEM"
    elif field in ("menu.price", "menu.unitprice", "menu.itemsubtotal",
                   "menu.sub_price", "menu.sub_unitprice"):
        return f"{prefix}MENU_PRICE"
    elif field.startswith("sub_total"):
        if "tax" in field:
            return f"{prefix}TAX"
        return f"{prefix}SUBTOTAL"
    elif field.startswith("total.total"):
        return f"{prefix}TOTAL"
    elif field.startswith("total"):
        return f"{prefix}OTHER"
    else:
        return f"{prefix}OTHER" if prefix else "O"


def parse_cord_ground_truth(gt_json: dict) -> list[dict]:
    """Parse CORD's ground truth JSON into a flat list of (text, label) entries.

    CORD GT format: {"gt_parse": {"menu": [{"nm": "...", "price": "..."}], ...}}
    """
    fields = []

    gt_parse = gt_json.get("gt_parse", gt_json)

    for group_name, items in gt_parse.items():
        if not isinstance(items, list):
            items = [items]

        for item in items:
            if isinstance(item, dict):
                for key, value in item.items():
                    if value and isinstance(value, str):
                        full_label = f"{group_name}.{key}"
                        fields.append({
                            "field_name": full_label,
                            "field_value": value.strip(),
                            "group": group_name,
                        })

    return fields


def process_cord_example(example) -> dict | None:
    """Process a single CORD example.

    Extracts text, bounding boxes, and maps labels for token classification.
    Also produces a structured field extraction format.
    """
    image = example.get("image")
    ground_truth = example.get("ground_truth")

    if not ground_truth:
        return None

    # Parse the ground truth (may be JSON string or dict)
    if isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
        except json.JSONDecodeError:
            return None
    else:
        gt = ground_truth

    # Extract structured fields
    fields = parse_cord_ground_truth(gt)
    if not fields:
        return None

    # Build a text representation from the fields
    text_parts = []
    for f in fields:
        text_parts.append(f"{f['field_name']}: {f['field_value']}")
    full_text = "\n".join(text_parts)

    # Also create field extraction format compatible with our pipeline
    extraction_fields = []
    for f in fields:
        extraction_fields.append({
            "field_name": f["field_name"],
            "field_value": f["field_value"],
            "field_type": "string",
            "confidence": 1.0,
            "extraction_method": "cord_ground_truth",
        })

    return {
        "text": full_text,
        "fields": extraction_fields,
        "field_count": len(fields),
        "groups": list(set(f["group"] for f in fields)),
    }


def load_cord(output_dir: str = None) -> DatasetDict:
    """Load CORD from HuggingFace and process for extraction training.

    Returns a DatasetDict with train/validation/test splits containing:
      - text: concatenated field text from receipt
      - fields: list of extracted field dicts
      - field_count: number of fields per receipt
    """
    print("Loading CORD dataset from HuggingFace...")
    ds = load_dataset("naver-clova-ix/cord-v2")

    processed_splits = {}
    for split_name in ds:
        print(f"Processing {split_name} split ({len(ds[split_name])} examples)...")
        processed = []
        for example in ds[split_name]:
            proc = process_cord_example(example)
            if proc:
                processed.append(proc)

        processed_splits[split_name] = Dataset.from_dict({
            "text": [p["text"] for p in processed],
            "fields": [json.dumps(p["fields"]) for p in processed],
            "field_count": [p["field_count"] for p in processed],
        })
        print(f"  {split_name}: {len(processed)} examples")

    dataset = DatasetDict(processed_splits)

    # Print stats
    total_fields = sum(
        p
        for split in dataset
        for p in dataset[split]["field_count"]
    )
    total_examples = sum(len(dataset[s]) for s in dataset)
    print(f"\nCORD statistics:")
    print(f"  Total examples: {total_examples}")
    print(f"  Total fields: {total_fields}")
    print(f"  Avg fields per receipt: {total_fields / max(total_examples, 1):.1f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        print(f"\nSaved to {output_dir}")

        metadata = {
            "dataset": "CORD",
            "labels": CORD_LABELS,
            "label2id": LABEL2ID,
            "num_labels": len(CORD_LABELS),
            "splits": {s: len(dataset[s]) for s in dataset},
            "total_fields": total_fields,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Load and process CORD dataset")
    parser.add_argument("--output", type=str, default="./data/cord_processed")
    args = parser.parse_args()

    load_cord(args.output)


if __name__ == "__main__":
    main()
