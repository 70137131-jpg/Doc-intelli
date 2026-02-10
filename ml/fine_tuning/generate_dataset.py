"""
Generate instruction-response training data for document intelligence tasks.
Supports three data sources:
  - synthetic: Generate via Gemini API (original behavior)
  - real:      Load from DocVQA + SQuAD 2.0 (HuggingFace)
  - hybrid:    Mix real data with synthetic augmentation

Output format: JSONL with Alpaca-style fields (instruction, input, output).

Usage:
    python generate_dataset.py --source real --output_dir ./data
    python generate_dataset.py --source hybrid --real_ratio 0.7 --output_dir ./data --num_per_task 200
    python generate_dataset.py --source synthetic --output_dir ./data --num_per_task 500
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import google.generativeai as genai
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Synthetic generation (original code)
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "summarization": {
        "system": "You are a data generator. Generate a realistic document text and its summary.",
        "templates": [
            {
                "instruction": "Summarize the following document in 2-3 sentences.",
                "context_prompt": "Generate a realistic {doc_type} document between 300-600 words. Topics: {topic}.",
                "output_prompt": "Now write a concise 2-3 sentence summary of the document you just generated.",
            },
            {
                "instruction": "Provide an executive summary of this document, highlighting key points.",
                "context_prompt": "Generate a realistic {doc_type} document between 400-800 words. Topics: {topic}.",
                "output_prompt": "Write an executive summary with 3-5 bullet points covering the key takeaways.",
            },
            {
                "instruction": "Extract the main conclusions from this document.",
                "context_prompt": "Generate a realistic {doc_type} with clear conclusions. Topics: {topic}.",
                "output_prompt": "List the 2-4 main conclusions from this document.",
            },
        ],
    },
    "extraction": {
        "system": "You are a data generator. Generate a realistic document and its extracted fields.",
        "templates": [
            {
                "instruction": "Extract all key fields from this invoice document as JSON.",
                "context_prompt": "Generate a realistic invoice with vendor name, invoice number, date, line items, subtotal, tax, and total amount.",
                "output_prompt": 'Return the extracted fields as JSON: {{"vendor_name": "...", "invoice_number": "...", "date": "...", "total_amount": "...", "tax": "...", "line_items": [...]}}',
            },
            {
                "instruction": "Extract the key information from this contract document.",
                "context_prompt": "Generate a realistic contract excerpt with parties, dates, terms, and obligations.",
                "output_prompt": 'Return extracted fields as JSON: {{"parties": [...], "effective_date": "...", "termination_date": "...", "key_terms": [...]}}',
            },
            {
                "instruction": "Parse this resume and extract structured information.",
                "context_prompt": "Generate a realistic resume with name, contact info, education, experience, and skills.",
                "output_prompt": 'Return extracted fields as JSON: {{"name": "...", "email": "...", "education": [...], "experience": [...], "skills": [...]}}',
            },
        ],
    },
    "classification": {
        "system": "You are a data generator. Generate a document and classify it.",
        "templates": [
            {
                "instruction": "Classify this document into one of: Invoice, Contract, Report, Resume, Letter, Other. Explain your reasoning.",
                "context_prompt": "Generate a realistic {doc_type} document between 200-400 words.",
                "output_prompt": 'Return JSON: {{"category": "{doc_type}", "confidence": 0.95, "reasoning": "This document is a {doc_type} because..."}}',
            },
        ],
    },
    "question_answering": {
        "system": "You are a data generator. Generate a document, a question about it, and the answer.",
        "templates": [
            {
                "instruction": "Answer the following question based on the provided document context.",
                "context_prompt": "Generate a realistic {doc_type} document (300-500 words) about {topic}. Then generate a specific factual question about it.",
                "output_prompt": "Provide a detailed answer citing specific information from the document. Use [Source: document, page 1] format.",
            },
            {
                "instruction": "Based on the document below, answer the question. If the answer is not in the document, say so.",
                "context_prompt": "Generate a realistic {doc_type} (200-400 words) about {topic}. Generate a question where the answer IS in the document.",
                "output_prompt": "Answer the question using only information from the document. Cite the relevant section.",
            },
        ],
    },
}

DOC_TYPES = ["Invoice", "Contract", "Report", "Resume", "Letter", "Memo", "Policy", "Manual"]
TOPICS = [
    "technology services", "healthcare", "financial planning", "software development",
    "marketing strategy", "legal compliance", "human resources", "supply chain",
    "real estate", "education", "manufacturing", "consulting services",
    "data analytics", "project management", "environmental sustainability",
]


def generate_single_sample(model, task_name: str, template: dict, doc_type: str, topic: str) -> dict | None:
    """Generate one training sample using Gemini."""
    context_prompt = template["context_prompt"].format(doc_type=doc_type, topic=topic)
    output_prompt = template["output_prompt"].format(doc_type=doc_type)

    prompt = f"""Generate a training example for an AI assistant.

Task: {task_name}

Step 1: {context_prompt}
Step 2: {output_prompt}

Return a JSON object with exactly these fields:
- "input": the generated document text (the context/document)
- "output": the response (summary, extracted fields, classification, or answer)

For question_answering tasks, also include:
- "question": the question about the document

Return ONLY valid JSON, no markdown formatting."""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)

        # Build Alpaca format
        instruction = template["instruction"]
        if task_name == "question_answering" and "question" in data:
            instruction = f"{instruction}\n\nQuestion: {data['question']}"

        return {
            "instruction": instruction,
            "input": data.get("input", ""),
            "output": data.get("output", ""),
            "task_type": task_name,
            "doc_type": doc_type,
            "source": "synthetic",
        }
    except Exception as e:
        return None


def quality_filter(sample: dict) -> tuple[bool, str]:
    """Filter out low-quality generated samples.

    Returns (passed, reason) tuple.
    """
    inp = sample.get("input", "")
    out = sample.get("output", "")

    # Min length checks
    if len(inp.split()) < 20:
        return False, "input_too_short"
    if len(out.split()) < 5:
        return False, "output_too_short"

    # Max length checks (avoid degenerate samples)
    if len(inp.split()) > 3000:
        return False, "input_too_long"

    # Check for placeholder / repetitive text
    low_inp = inp.lower()
    if "[insert" in low_inp or "lorem ipsum" in low_inp:
        return False, "placeholder_text"

    # Check input != output (degenerate copy)
    if inp.strip() == out.strip():
        return False, "input_equals_output"

    # Check for valid JSON in extraction tasks
    task_type = sample.get("task_type", "")
    if task_type == "extraction":
        if "{" not in out and "[" not in out:
            return False, "extraction_missing_json"

    return True, "ok"


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """Remove near-duplicate samples based on output text hash."""
    seen_hashes = set()
    unique = []
    for s in samples:
        h = hash(s.get("output", "")[:200].lower().strip())
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(s)
    return unique


def generate_synthetic_data(api_key: str, num_per_task: int) -> list[dict]:
    """Generate synthetic training data via Gemini (original behavior)."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    all_samples = []
    filter_stats = {"total_generated": 0, "passed": 0, "rejected": {}}

    for task_name, task_config in TASK_PROMPTS.items():
        templates = task_config["templates"]
        samples_per_template = num_per_task // len(templates)

        print(f"\n{'=' * 60}")
        print(f"Generating {num_per_task} samples for task: {task_name}")
        print(f"  {len(templates)} templates x {samples_per_template} samples each")

        task_samples = []
        for template in templates:
            for i in range(samples_per_template):
                doc_type = random.choice(DOC_TYPES)
                topic = random.choice(TOPICS)

                sample = generate_single_sample(model, task_name, template, doc_type, topic)
                filter_stats["total_generated"] += 1

                if sample and sample["input"] and sample["output"]:
                    passed, reason = quality_filter(sample)
                    if passed:
                        task_samples.append(sample)
                        filter_stats["passed"] += 1
                    else:
                        filter_stats["rejected"][reason] = filter_stats["rejected"].get(reason, 0) + 1

                if (i + 1) % 25 == 0:
                    print(f"  [{task_name}] Generated {i + 1}/{samples_per_template} for template")

                # Rate limiting
                time.sleep(0.5)

        task_samples = deduplicate_samples(task_samples)
        all_samples.extend(task_samples)
        print(f"  Total for {task_name}: {len(task_samples)} samples (after filtering)")

    # Print quality filter report
    print(f"\n{'=' * 60}")
    print("Synthetic Data Quality Report:")
    print(f"  Total generated: {filter_stats['total_generated']}")
    print(f"  Passed filters:  {filter_stats['passed']}")
    for reason, count in sorted(filter_stats["rejected"].items(), key=lambda x: -x[1]):
        print(f"  Rejected ({reason}): {count}")

    return all_samples


# ---------------------------------------------------------------------------
# Real dataset loaders: DocVQA + SQuAD 2.0
# ---------------------------------------------------------------------------

QA_INSTRUCTIONS = [
    "Answer the following question based on the provided document context.",
    "Based on the document below, answer the question. If the answer is not in the document, say so.",
    "Using only the information in the provided text, answer this question.",
    "Read the document and answer the question with specific details from the text.",
]

UNANSWERABLE_RESPONSES = [
    "The document does not contain enough information to answer this question.",
    "Based on the provided document, this question cannot be answered.",
    "The answer to this question is not found in the given text.",
]


def load_docvqa_data(max_samples: int = 10000) -> list[dict]:
    """Load DocVQA dataset and convert to Alpaca instruction format.

    DocVQA contains document images with questions and answers.
    We use the text-based question-answer pairs.
    """
    print(f"\nLoading DocVQA from HuggingFace (max {max_samples} samples)...")

    try:
        ds = load_dataset("lmms-lab/DocVQA", split="train", streaming=True)
    except Exception:
        # Fallback to alternative DocVQA source
        try:
            ds = load_dataset("eliolio/DocVQA", split="train", streaming=True)
        except Exception as e:
            print(f"  Warning: Could not load DocVQA: {e}")
            return []

    samples = []
    for example in ds:
        if len(samples) >= max_samples:
            break

        question = example.get("question", "")
        answers = example.get("answers", example.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]

        if not question or not answers:
            continue

        # Use the first answer (or most common if multiple)
        answer = answers[0] if answers else ""
        if not answer:
            continue

        # DocVQA may have OCR text or we need to describe the context
        context = example.get("context", example.get("ocr_text", ""))
        if not context:
            # Build context from question + answer if no text available
            context = f"[Document context for question: {question}]"

        instruction = random.choice(QA_INSTRUCTIONS)
        instruction += f"\n\nQuestion: {question}"

        samples.append({
            "instruction": instruction,
            "input": context,
            "output": answer,
            "task_type": "question_answering",
            "doc_type": "document",
            "source": "docvqa",
        })

        if len(samples) % 1000 == 0:
            print(f"  Loaded {len(samples)} DocVQA samples...")

    print(f"  DocVQA: {len(samples)} samples loaded")
    return samples


def load_squad_data(max_samples: int = 15000, include_unanswerable: bool = True) -> list[dict]:
    """Load SQuAD 2.0 dataset and convert to Alpaca instruction format.

    SQuAD 2.0 includes both answerable and unanswerable questions.
    Unanswerable questions train the model to say "I don't know" — crucial
    for reducing hallucination in RAG.
    """
    print(f"\nLoading SQuAD 2.0 from HuggingFace (max {max_samples} samples)...")
    ds = load_dataset("rajpurkar/squad_v2", split="train")

    # Separate answerable and unanswerable
    answerable = []
    unanswerable = []

    for example in ds:
        answers = example.get("answers", {})
        answer_texts = answers.get("text", [])
        context = example.get("context", "")
        question = example.get("question", "")

        if not context or not question:
            continue

        if answer_texts:
            answerable.append({
                "question": question,
                "context": context,
                "answer": answer_texts[0],
                "is_answerable": True,
            })
        elif include_unanswerable:
            unanswerable.append({
                "question": question,
                "context": context,
                "answer": "",
                "is_answerable": False,
            })

    # Balance: 80% answerable, 20% unanswerable
    random.shuffle(answerable)
    random.shuffle(unanswerable)

    n_answerable = int(max_samples * 0.8)
    n_unanswerable = max_samples - n_answerable

    selected = answerable[:n_answerable] + unanswerable[:n_unanswerable]
    random.shuffle(selected)

    samples = []
    for item in selected:
        instruction = random.choice(QA_INSTRUCTIONS)
        instruction += f"\n\nQuestion: {item['question']}"

        if item["is_answerable"]:
            output = item["answer"]
        else:
            output = random.choice(UNANSWERABLE_RESPONSES)

        samples.append({
            "instruction": instruction,
            "input": item["context"],
            "output": output,
            "task_type": "question_answering",
            "doc_type": "article",
            "source": "squad_v2",
        })

    print(f"  SQuAD 2.0: {len(samples)} samples ({sum(1 for s in selected if s['is_answerable'])} answerable, "
          f"{sum(1 for s in selected if not s['is_answerable'])} unanswerable)")
    return samples


def load_squad_summarization(max_samples: int = 3000) -> list[dict]:
    """Convert SQuAD contexts into extractive summarization training data.

    Uses the first sentence of each context as a simple extractive summary.
    """
    print(f"\nGenerating summarization data from SQuAD contexts (max {max_samples})...")
    ds = load_dataset("rajpurkar/squad_v2", split="train")

    # Collect unique contexts (SQuAD reuses contexts for multiple questions)
    seen_contexts = set()
    unique_contexts = []
    for example in ds:
        ctx = example.get("context", "")
        if ctx and ctx not in seen_contexts and len(ctx.split()) > 50:
            seen_contexts.add(ctx)
            unique_contexts.append(ctx)
            if len(unique_contexts) >= max_samples:
                break

    random.shuffle(unique_contexts)

    samples = []
    summarization_instructions = [
        "Summarize the following document in 2-3 sentences.",
        "Provide a brief summary of the key points in this text.",
        "What are the main ideas presented in this passage?",
    ]

    for context in unique_contexts[:max_samples]:
        # Extractive summary: first 2 sentences
        sentences = context.replace("! ", ".\n").replace("? ", ".\n").split(". ")
        summary = ". ".join(sentences[:2]).strip()
        if not summary.endswith("."):
            summary += "."

        samples.append({
            "instruction": random.choice(summarization_instructions),
            "input": context,
            "output": summary,
            "task_type": "summarization",
            "doc_type": "article",
            "source": "squad_v2_summarization",
        })

    print(f"  Summarization from SQuAD: {len(samples)} samples")
    return samples


def load_real_data(max_qa_samples: int = 15000, max_summarization_samples: int = 3000) -> list[dict]:
    """Load all real datasets and combine."""
    all_samples = []

    # DocVQA — document-specific QA
    docvqa_samples = load_docvqa_data(max_samples=max_qa_samples // 2)
    all_samples.extend(docvqa_samples)

    # SQuAD 2.0 — general QA (with unanswerable questions)
    squad_samples = load_squad_data(max_samples=max_qa_samples // 2)
    all_samples.extend(squad_samples)

    # SQuAD contexts as summarization data
    summ_samples = load_squad_summarization(max_samples=max_summarization_samples)
    all_samples.extend(summ_samples)

    print(f"\n{'=' * 60}")
    print(f"Total real samples: {len(all_samples)}")
    print(f"  Source distribution: {dict(Counter(s['source'] for s in all_samples))}")
    print(f"  Task distribution:  {dict(Counter(s['task_type'] for s in all_samples))}")

    return all_samples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def save_dataset(all_samples: list[dict], output_dir: str):
    """Shuffle, split, and save as JSONL."""
    # Apply quality filter and dedup
    filtered = []
    filter_stats = {"total": len(all_samples), "passed": 0, "rejected": {}}
    for s in all_samples:
        passed, reason = quality_filter(s)
        if passed:
            filtered.append(s)
            filter_stats["passed"] += 1
        else:
            filter_stats["rejected"][reason] = filter_stats["rejected"].get(reason, 0) + 1

    filtered = deduplicate_samples(filtered)

    print(f"\nAfter filtering + dedup: {len(filtered)} samples (from {len(all_samples)})")
    if filter_stats["rejected"]:
        for reason, count in sorted(filter_stats["rejected"].items(), key=lambda x: -x[1]):
            print(f"  Rejected ({reason}): {count}")

    # Shuffle and split
    random.shuffle(filtered)
    n = len(filtered)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": filtered[:train_end],
        "val": filtered[train_end:val_end],
        "test": filtered[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_data in splits.items():
        filepath = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(filepath, "w") as f:
            for sample in split_data:
                f.write(json.dumps(sample) + "\n")
        print(f"\n{split_name}: {len(split_data)} samples → {filepath}")

    # Save metadata
    metadata = {
        "total_samples": n,
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "tasks": {
            task: len([s for s in filtered if s["task_type"] == task])
            for task in set(s["task_type"] for s in filtered)
        },
        "sources": dict(Counter(s.get("source", "unknown") for s in filtered)),
        "quality_filter": filter_stats,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTotal: {n} samples across {len(metadata['tasks'])} tasks")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Generate fine-tuning dataset")
    parser.add_argument("--source", type=str, default="synthetic",
                        choices=["synthetic", "real", "hybrid"],
                        help="Data source: synthetic (Gemini), real (DocVQA+SQuAD), or hybrid")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--num_per_task", type=int, default=500,
                        help="Samples per task for synthetic generation")
    parser.add_argument("--max_qa_samples", type=int, default=15000,
                        help="Max QA samples from real datasets")
    parser.add_argument("--max_summarization_samples", type=int, default=3000,
                        help="Max summarization samples from SQuAD contexts")
    parser.add_argument("--real_ratio", type=float, default=0.7,
                        help="Ratio of real data in hybrid mode (0.0-1.0)")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="Generate fewer samples for testing")
    args = parser.parse_args()

    if args.fast:
        args.num_per_task = 20
        args.max_qa_samples = 200
        args.max_summarization_samples = 50

    all_samples = []

    if args.source in ("real", "hybrid"):
        real_samples = load_real_data(
            max_qa_samples=args.max_qa_samples,
            max_summarization_samples=args.max_summarization_samples,
        )
        all_samples.extend(real_samples)

    if args.source in ("synthetic", "hybrid"):
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            if args.source == "synthetic":
                print("Error: Set GEMINI_API_KEY environment variable or pass --api_key")
                exit(1)
            else:
                print("Warning: No API key — skipping synthetic data for hybrid mode.")
        else:
            # In hybrid mode, scale synthetic samples based on real_ratio
            if args.source == "hybrid" and all_samples:
                target_synthetic = int(len(all_samples) * (1 - args.real_ratio) / args.real_ratio)
                num_tasks = len(TASK_PROMPTS)
                adjusted_per_task = max(10, target_synthetic // num_tasks)
                print(f"\nHybrid mode: targeting {target_synthetic} synthetic samples "
                      f"({adjusted_per_task} per task) to achieve {args.real_ratio:.0%} real ratio")
            else:
                adjusted_per_task = args.num_per_task

            synthetic_samples = generate_synthetic_data(api_key, adjusted_per_task)
            all_samples.extend(synthetic_samples)

    if all_samples:
        save_dataset(all_samples, args.output_dir)
    else:
        print("No samples generated or loaded.")


if __name__ == "__main__":
    main()
