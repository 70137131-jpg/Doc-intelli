"""
Evaluate the trained document classifier.

Supports two modes:
  - Standard: evaluate on local test split (default)
  - Benchmark: evaluate on RVL-CDIP's official test split with per-class metrics

Usage:
    python evaluate_classifier.py --model_dir ./output/model --data_dir ./data/dataset
    python evaluate_classifier.py --model_dir ./output/model --benchmark --max_samples 1000
"""

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


CATEGORIES = ["Invoice", "Contract", "Report", "Resume", "Letter", "Other"]

# Same mapping used in prepare_dataset.py
RVL_CDIP_CLASS_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "questionnaire", "resume", "memo", "presentation",
]

RVL_CDIP_TO_CATEGORY = {
    "letter": "Letter",
    "form": "Other",
    "email": "Letter",
    "handwritten": "Other",
    "advertisement": "Other",
    "scientific_report": "Report",
    "scientific_publication": "Report",
    "specification": "Other",
    "file_folder": "Other",
    "news_article": "Report",
    "budget": "Other",
    "invoice": "Invoice",
    "questionnaire": "Other",
    "resume": "Resume",
    "memo": "Letter",
    "presentation": "Other",
}


def evaluate_on_local(classifier, data_dir: str) -> dict:
    """Evaluate on the local test split."""
    dataset = load_from_disk(data_dir)
    test_data = dataset["test"]

    print(f"Evaluating on {len(test_data)} local test samples...")

    predictions = []
    true_labels = []
    confidences = []

    for sample in test_data:
        text = sample["text"][:512]
        true_label = sample["label"]

        result = classifier(text)[0]
        pred_label = max(range(len(result)), key=lambda i: result[i]["score"])
        pred_confidence = result[pred_label]["score"]

        predictions.append(pred_label)
        true_labels.append(true_label)
        confidences.append(pred_confidence)

    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average="weighted")
    report = classification_report(
        true_labels, predictions, target_names=CATEGORIES, output_dict=True
    )
    cm = confusion_matrix(true_labels, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=CATEGORIES))
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nAverage Confidence: {np.mean(confidences):.4f}")

    return {
        "mode": "local",
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "avg_confidence": float(np.mean(confidences)),
        "num_test_samples": len(test_data),
    }


def evaluate_on_rvl_cdip(classifier, max_samples: int = 1000) -> dict:
    """Evaluate on RVL-CDIP's official test split.

    Reports both per-original-class (16-class) and per-mapped-category (6-class) metrics.
    Uses OCR to extract text from test images.
    """
    import pytesseract

    print(f"\nLoading RVL-CDIP test split from HuggingFace (max {max_samples} samples)...")
    ds = load_dataset("aharley/rvl_cdip", split="test", streaming=True)

    predictions = []
    true_labels = []
    true_rvl_classes = []
    confidences = []
    skipped = 0

    for example in ds:
        if len(predictions) >= max_samples:
            break

        rvl_label_idx = example["label"]
        rvl_class_name = RVL_CDIP_CLASS_NAMES[rvl_label_idx]
        true_category = RVL_CDIP_TO_CATEGORY[rvl_class_name]
        true_label = CATEGORIES.index(true_category)

        # OCR the image
        try:
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            text = pytesseract.image_to_string(image, lang="eng").strip()
        except Exception:
            skipped += 1
            continue

        if len(text) < 30:
            skipped += 1
            continue

        # Classify
        result = classifier(text[:512])[0]
        pred_label = max(range(len(result)), key=lambda i: result[i]["score"])
        pred_confidence = result[pred_label]["score"]

        predictions.append(pred_label)
        true_labels.append(true_label)
        true_rvl_classes.append(rvl_class_name)
        confidences.append(pred_confidence)

        if len(predictions) % 100 == 0:
            print(f"  Processed {len(predictions)}/{max_samples} (skipped {skipped})...")

    print(f"\nRVL-CDIP Benchmark Results ({len(predictions)} samples, {skipped} skipped):")

    # 6-category metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_weighted = f1_score(true_labels, predictions, average="weighted")
    report = classification_report(
        true_labels, predictions, target_names=CATEGORIES, output_dict=True
    )
    cm = confusion_matrix(true_labels, predictions)

    print(f"\n6-Category Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(classification_report(true_labels, predictions, target_names=CATEGORIES))

    # Per-original-RVL-CDIP-class accuracy
    print(f"\nPer-RVL-CDIP-Class Accuracy (mapped to our categories):")
    rvl_class_correct = {}
    rvl_class_total = {}
    for rvl_class, true_l, pred_l in zip(true_rvl_classes, true_labels, predictions):
        rvl_class_total[rvl_class] = rvl_class_total.get(rvl_class, 0) + 1
        if true_l == pred_l:
            rvl_class_correct[rvl_class] = rvl_class_correct.get(rvl_class, 0) + 1

    per_class_accuracy = {}
    for cls in sorted(rvl_class_total.keys()):
        correct = rvl_class_correct.get(cls, 0)
        total = rvl_class_total[cls]
        acc = correct / total if total > 0 else 0
        mapped = RVL_CDIP_TO_CATEGORY[cls]
        per_class_accuracy[cls] = {"accuracy": acc, "mapped_to": mapped, "count": total}
        print(f"  {cls:25s} -> {mapped:10s}  acc={acc:.3f}  (n={total})")

    return {
        "mode": "benchmark_rvl_cdip",
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "avg_confidence": float(np.mean(confidences)),
        "num_test_samples": len(predictions),
        "num_skipped": skipped,
        "per_rvl_class_accuracy": per_class_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate document classifier")
    parser.add_argument("--model_dir", type=str, default="./output/model")
    parser.add_argument("--data_dir", type=str, default="./data/dataset")
    parser.add_argument("--output_dir", type=str, default="./output/results")
    parser.add_argument("--benchmark", action="store_true",
                        help="Evaluate on RVL-CDIP official test split (requires pytesseract)")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max samples for benchmark evaluation")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
    )

    results = {}

    # Standard local evaluation
    results["local"] = evaluate_on_local(classifier, args.data_dir)

    # Benchmark evaluation on RVL-CDIP test split
    if args.benchmark:
        results["benchmark"] = evaluate_on_rvl_cdip(classifier, max_samples=args.max_samples)

        # Print comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON: Local vs RVL-CDIP Benchmark")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<20s} {'Local':>10s} {'Benchmark':>10s}")
        print(f"  {'Accuracy':<20s} {results['local']['accuracy']:>10.4f} {results['benchmark']['accuracy']:>10.4f}")
        print(f"  {'F1 (weighted)':<20s} {results['local']['f1_weighted']:>10.4f} {results['benchmark']['f1_weighted']:>10.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/evaluation_results.json")


if __name__ == "__main__":
    main()
