"""
Prepare training dataset for document classification.
Supports three data sources:
  - synthetic: Generate via Gemini API (original behavior)
  - rvl_cdip:  Load real documents from RVL-CDIP (HuggingFace) with OCR
  - hybrid:    Mix real RVL-CDIP data with synthetic augmentation

Usage:
    python prepare_dataset.py --source rvl_cdip --max_samples 10000 --output ./data/dataset
    python prepare_dataset.py --source synthetic --num_samples 100 --output ./data/dataset
    python prepare_dataset.py --source hybrid --max_samples 8000 --num_samples 50 --output ./data/dataset
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import google.generativeai as genai
from datasets import Dataset, DatasetDict, load_dataset

CATEGORIES = ["Invoice", "Contract", "Report", "Resume", "Letter", "Other"]

# RVL-CDIP has 16 classes (0-15). Map them to our 6 categories.
# RVL-CDIP labels: letter, form, email, handwritten, advertisement, scientific_report,
#   scientific_publication, specification, file_folder, news_article, budget, invoice,
#   questionnaire, resume, memo, presentation
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

PROMPTS = {
    "Invoice": "Generate a realistic invoice document text. Include vendor name, invoice number, date, line items with descriptions and amounts, subtotal, tax, and total. Vary the format and content.",
    "Contract": "Generate a realistic contract or agreement text. Include parties involved, effective date, terms and conditions, obligations, termination clauses, and signatures section. Vary the type (employment, service, NDA, lease).",
    "Report": "Generate a realistic business or technical report text. Include title, executive summary, findings, data analysis, conclusions, and recommendations. Vary the topic and format.",
    "Resume": "Generate a realistic resume/CV text. Include name, contact info, summary, work experience with dates and descriptions, education, skills, and certifications. Vary the career level and industry.",
    "Letter": "Generate a realistic business letter text. Include sender, recipient, date, subject, body paragraphs, and closing. Vary the purpose (complaint, inquiry, cover letter, recommendation).",
    "Other": "Generate a realistic miscellaneous document text that doesn't fit into invoice, contract, report, resume, or letter categories. This could be a memo, meeting notes, policy document, manual, or FAQ.",
}


def ocr_image(image) -> str:
    """Extract text from a PIL image using pytesseract."""
    import pytesseract

    # Convert to RGB if needed (RVL-CDIP images are grayscale tiffs)
    if image.mode != "RGB":
        image = image.convert("RGB")

    text = pytesseract.image_to_string(image, lang="eng")
    return text.strip()


def load_rvl_cdip_data(max_samples: int = 10000, min_text_length: int = 50) -> list[dict]:
    """Load and OCR documents from the RVL-CDIP dataset.

    Downloads from HuggingFace, applies OCR to extract text, and maps
    the 16 RVL-CDIP classes to our 6 categories.
    """
    print("Loading RVL-CDIP dataset from HuggingFace...")
    ds = load_dataset("aharley/rvl_cdip", split="train", streaming=True)

    # Calculate per-category budget for balanced sampling
    per_category_budget = max_samples // len(CATEGORIES)
    category_counts = defaultdict(int)
    samples = []
    skipped_ocr = 0
    processed = 0

    print(f"Target: {max_samples} samples ({per_category_budget} per category)")
    print("Running OCR on document images (this may take a while)...")

    for example in ds:
        rvl_label_idx = example["label"]
        rvl_class_name = RVL_CDIP_CLASS_NAMES[rvl_label_idx]
        category = RVL_CDIP_TO_CATEGORY[rvl_class_name]

        # Skip if this category is already full
        if category_counts[category] >= per_category_budget:
            # Check if all categories are full
            if all(category_counts[c] >= per_category_budget for c in CATEGORIES):
                break
            continue

        # OCR the image
        try:
            text = ocr_image(example["image"])
        except Exception as e:
            skipped_ocr += 1
            continue

        # Skip if OCR produced too little text
        if len(text) < min_text_length:
            skipped_ocr += 1
            continue

        samples.append({
            "text": text,
            "label": CATEGORIES.index(category),
            "category": category,
            "source": "rvl_cdip",
            "rvl_class": rvl_class_name,
        })
        category_counts[category] += 1
        processed += 1

        if processed % 100 == 0:
            dist = {c: category_counts[c] for c in CATEGORIES}
            print(f"  Processed {processed} samples (skipped {skipped_ocr} OCR failures) — {dist}")

    print(f"\nRVL-CDIP loading complete:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Skipped (OCR failures/short text): {skipped_ocr}")
    print(f"  Category distribution: {dict(Counter(s['category'] for s in samples))}")
    return samples


def generate_synthetic_data(api_key: str, num_per_category: int = 100) -> list[dict]:
    """Generate synthetic documents using Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    samples = []

    for category, prompt in PROMPTS.items():
        print(f"Generating {num_per_category} samples for {category}...")
        for i in range(num_per_category):
            try:
                full_prompt = f"""{prompt}

Generate a unique document (sample {i+1}/{num_per_category}).
Return ONLY the document text, no explanations or metadata.
Make it realistic and between 200-800 words."""

                response = model.generate_content(full_prompt)
                text = response.text.strip()

                if text:
                    samples.append({
                        "text": text,
                        "label": CATEGORIES.index(category),
                        "category": category,
                        "source": "synthetic",
                    })

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{num_per_category}")

            except Exception as e:
                print(f"  Error generating sample {i+1}: {e}")
                continue

    return samples


def create_dataset(samples: list[dict], output_dir: str) -> DatasetDict:
    """Create a Hugging Face DatasetDict from samples."""
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    def to_dataset(data):
        return Dataset.from_dict({
            "text": [s["text"] for s in data],
            "label": [s["label"] for s in data],
            "category": [s["category"] for s in data],
        })

    dataset = DatasetDict({
        "train": to_dataset(train_data),
        "validation": to_dataset(val_data),
        "test": to_dataset(test_data),
    })

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"\nDataset saved to {output_dir}")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Save as JSON for inspection
    with open(os.path.join(output_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)

    # Save metadata
    metadata = {
        "total_samples": n,
        "splits": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
        "category_distribution": dict(Counter(s["category"] for s in samples)),
        "source_distribution": dict(Counter(s.get("source", "unknown") for s in samples)),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare classification dataset")
    parser.add_argument("--source", type=str, default="synthetic",
                        choices=["synthetic", "rvl_cdip", "hybrid"],
                        help="Data source: synthetic (Gemini), rvl_cdip (real), or hybrid (both)")
    parser.add_argument("--output", type=str, default="./data/dataset")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Samples per category for synthetic generation")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max samples to load from RVL-CDIP (balanced across categories)")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
    args = parser.parse_args()

    samples = []

    if args.source in ("rvl_cdip", "hybrid"):
        real_samples = load_rvl_cdip_data(max_samples=args.max_samples)
        samples.extend(real_samples)
        print(f"\nLoaded {len(real_samples)} real samples from RVL-CDIP")

    if args.source in ("synthetic", "hybrid"):
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            if args.source == "synthetic":
                print("Error: GEMINI_API_KEY not set. Provide via --api_key or environment variable.")
                return
            else:
                print("Warning: No API key — skipping synthetic augmentation for hybrid mode.")
        else:
            synthetic_samples = generate_synthetic_data(api_key, args.num_samples)
            samples.extend(synthetic_samples)
            print(f"\nGenerated {len(synthetic_samples)} synthetic samples")

    if samples:
        print(f"\nTotal samples: {len(samples)}")
        create_dataset(samples, args.output)
    else:
        print("No samples generated or loaded.")


if __name__ == "__main__":
    main()
