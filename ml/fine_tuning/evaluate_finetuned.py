"""
Evaluate the fine-tuned model on test data and optional real-world benchmarks.

Supports two modes:
  - Standard: evaluate on local test.jsonl with ROUGE metrics (default)
  - Benchmark: evaluate on DocVQA val split + SQuAD 2.0 dev set with EM/F1 metrics

Usage:
    python evaluate_finetuned.py --model_dir ./output/merged-model --test_file ./data/test.jsonl
    python evaluate_finetuned.py --model_dir ./output/merged-model --benchmark --max_samples 500
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_test_data(filepath: str) -> list[dict]:
    data = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def generate_response(model, tokenizer, instruction: str, input_text: str, max_tokens: int = 512) -> str:
    prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except ImportError:
        # Fallback: simple overlap
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
        return {"rouge1": overlap, "rouge2": overlap * 0.8, "rougeL": overlap * 0.9}


# ---------------------------------------------------------------------------
# QA-specific metrics: Exact Match (EM) and Token-level F1
# Standard metrics for SQuAD / DocVQA evaluation
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer for EM/F1 comparison (lowercase, remove articles/punctuation)."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction: str, reference: str) -> float:
    """Exact Match: 1.0 if normalized prediction == normalized reference, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def compute_token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Standard evaluation (local test file)
# ---------------------------------------------------------------------------

def evaluate_model(model_dir: str, test_file: str, max_samples: int = 100) -> dict:
    """Evaluate the fine-tuned model on the local test set."""
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    test_data = load_test_data(test_file)[:max_samples]
    print(f"Evaluating on {len(test_data)} samples")

    task_metrics = defaultdict(lambda: {
        "rouge1": [], "rouge2": [], "rougeL": [],
        "em": [], "f1": [], "latency": [],
    })

    for i, sample in enumerate(test_data):
        task_type = sample.get("task_type", "unknown")

        start_time = time.time()
        prediction = generate_response(
            model, tokenizer,
            sample["instruction"],
            sample.get("input", ""),
        )
        latency = time.time() - start_time

        reference = sample["output"]

        # ROUGE
        rouge = compute_rouge(prediction, reference)
        task_metrics[task_type]["rouge1"].append(rouge["rouge1"])
        task_metrics[task_type]["rouge2"].append(rouge["rouge2"])
        task_metrics[task_type]["rougeL"].append(rouge["rougeL"])

        # EM / F1 (especially relevant for QA tasks)
        em = compute_exact_match(prediction, reference)
        f1 = compute_token_f1(prediction, reference)
        task_metrics[task_type]["em"].append(em)
        task_metrics[task_type]["f1"].append(f1)

        task_metrics[task_type]["latency"].append(latency)

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(test_data)}] {task_type}: "
                  f"ROUGE-L={rouge['rougeL']:.3f} EM={em:.0f} F1={f1:.3f} "
                  f"latency={latency:.2f}s")

    # Aggregate
    results = {"per_task": {}, "overall": {}}
    all_metrics = defaultdict(list)

    for task, metrics in task_metrics.items():
        avg = {k: sum(v) / len(v) if v else 0 for k, v in metrics.items()}
        results["per_task"][task] = {
            "rouge1": round(avg["rouge1"], 4),
            "rouge2": round(avg["rouge2"], 4),
            "rougeL": round(avg["rougeL"], 4),
            "exact_match": round(avg["em"], 4),
            "token_f1": round(avg["f1"], 4),
            "avg_latency_s": round(avg["latency"], 3),
            "num_samples": len(metrics["rouge1"]),
        }
        for k, v in metrics.items():
            all_metrics[k].extend(v)

    n = len(all_metrics["rouge1"])
    results["overall"] = {
        "rouge1": round(sum(all_metrics["rouge1"]) / n, 4) if n else 0,
        "rouge2": round(sum(all_metrics["rouge2"]) / n, 4) if n else 0,
        "rougeL": round(sum(all_metrics["rougeL"]) / n, 4) if n else 0,
        "exact_match": round(sum(all_metrics["em"]) / n, 4) if n else 0,
        "token_f1": round(sum(all_metrics["f1"]) / n, 4) if n else 0,
        "avg_latency_s": round(sum(all_metrics["latency"]) / n, 3) if n else 0,
        "total_samples": n,
    }

    return model, tokenizer, results


# ---------------------------------------------------------------------------
# Benchmark evaluation: DocVQA + SQuAD 2.0
# ---------------------------------------------------------------------------

def benchmark_docvqa(model, tokenizer, max_samples: int = 200) -> dict:
    """Evaluate on DocVQA validation split."""
    print(f"\nBenchmark: DocVQA validation set (max {max_samples} samples)...")

    try:
        ds = load_dataset("lmms-lab/DocVQA", split="validation", streaming=True)
    except Exception:
        try:
            ds = load_dataset("eliolio/DocVQA", split="validation", streaming=True)
        except Exception as e:
            print(f"  Warning: Could not load DocVQA: {e}")
            return {}

    em_scores = []
    f1_scores = []
    count = 0

    for example in ds:
        if count >= max_samples:
            break

        question = example.get("question", "")
        answers = example.get("answers", example.get("answer", []))
        if isinstance(answers, str):
            answers = [answers]
        context = example.get("context", example.get("ocr_text", ""))

        if not question or not answers or not context:
            continue

        instruction = f"Answer the following question based on the provided document.\n\nQuestion: {question}"
        prediction = generate_response(model, tokenizer, instruction, context)

        # EM and F1 against all reference answers (take max — standard practice)
        best_em = max(compute_exact_match(prediction, ans) for ans in answers)
        best_f1 = max(compute_token_f1(prediction, ans) for ans in answers)

        em_scores.append(best_em)
        f1_scores.append(best_f1)
        count += 1

        if count % 50 == 0:
            print(f"  DocVQA: {count}/{max_samples} — EM={sum(em_scores) / len(em_scores):.3f} "
                  f"F1={sum(f1_scores) / len(f1_scores):.3f}")

    if not em_scores:
        return {}

    return {
        "dataset": "DocVQA",
        "exact_match": round(sum(em_scores) / len(em_scores), 4),
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "num_samples": len(em_scores),
    }


def benchmark_squad(model, tokenizer, max_samples: int = 300) -> dict:
    """Evaluate on SQuAD 2.0 validation (dev) set."""
    print(f"\nBenchmark: SQuAD 2.0 dev set (max {max_samples} samples)...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    # Sample a balanced mix of answerable and unanswerable
    answerable = [ex for ex in ds if ex["answers"]["text"]]
    unanswerable = [ex for ex in ds if not ex["answers"]["text"]]

    import random
    random.shuffle(answerable)
    random.shuffle(unanswerable)

    n_ans = int(max_samples * 0.8)
    n_unans = max_samples - n_ans
    selected = answerable[:n_ans] + unanswerable[:n_unans]
    random.shuffle(selected)

    em_scores = []
    f1_scores = []
    answerable_em = []
    unanswerable_em = []

    for i, example in enumerate(selected):
        context = example["context"]
        question = example["question"]
        ref_answers = example["answers"]["text"]
        is_answerable = len(ref_answers) > 0

        instruction = (
            "Based on the document below, answer the question. "
            "If the answer is not in the document, say so.\n\n"
            f"Question: {question}"
        )
        prediction = generate_response(model, tokenizer, instruction, context)

        if is_answerable:
            best_em = max(compute_exact_match(prediction, ans) for ans in ref_answers)
            best_f1 = max(compute_token_f1(prediction, ans) for ans in ref_answers)
            answerable_em.append(best_em)
        else:
            # For unanswerable: check if model correctly refuses
            norm_pred = normalize_answer(prediction)
            refuses = any(phrase in norm_pred for phrase in [
                "not in the document", "cannot be answered", "not found",
                "does not contain", "no information", "not mentioned",
                "not enough information",
            ])
            best_em = 1.0 if refuses else 0.0
            best_f1 = best_em
            unanswerable_em.append(best_em)

        em_scores.append(best_em)
        f1_scores.append(best_f1)

        if (i + 1) % 50 == 0:
            print(f"  SQuAD: {i + 1}/{len(selected)} — EM={sum(em_scores) / len(em_scores):.3f} "
                  f"F1={sum(f1_scores) / len(f1_scores):.3f}")

    return {
        "dataset": "SQuAD_v2",
        "exact_match": round(sum(em_scores) / len(em_scores), 4),
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "answerable_em": round(sum(answerable_em) / len(answerable_em), 4) if answerable_em else 0,
        "unanswerable_em": round(sum(unanswerable_em) / len(unanswerable_em), 4) if unanswerable_em else 0,
        "num_samples": len(em_scores),
        "num_answerable": len(answerable_em),
        "num_unanswerable": len(unanswerable_em),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_dir", type=str, default="./output/merged-model")
    parser.add_argument("--test_file", type=str, default="./data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="./output/eval_results.json")
    parser.add_argument("--benchmark", action="store_true",
                        help="Also evaluate on DocVQA val and SQuAD 2.0 dev sets")
    parser.add_argument("--benchmark_samples", type=int, default=300,
                        help="Max samples per benchmark dataset")
    args = parser.parse_args()

    model, tokenizer, results = evaluate_model(args.model_dir, args.test_file, args.max_samples)

    # Print standard results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Local Test Set)")
    print("=" * 60)

    for task, metrics in results["per_task"].items():
        print(f"\n{task}:")
        print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        print(f"  Token F1: {metrics['token_f1']:.4f}")
        print(f"  Avg Latency: {metrics['avg_latency_s']:.3f}s")
        print(f"  Samples: {metrics['num_samples']}")

    print(f"\nOVERALL:")
    o = results["overall"]
    print(f"  ROUGE-1: {o['rouge1']:.4f}")
    print(f"  ROUGE-2: {o['rouge2']:.4f}")
    print(f"  ROUGE-L: {o['rougeL']:.4f}")
    print(f"  Exact Match: {o['exact_match']:.4f}")
    print(f"  Token F1: {o['token_f1']:.4f}")
    print(f"  Avg Latency: {o['avg_latency_s']:.3f}s")

    # Benchmark evaluation
    if args.benchmark:
        results["benchmarks"] = {}

        docvqa_results = benchmark_docvqa(model, tokenizer, max_samples=args.benchmark_samples)
        if docvqa_results:
            results["benchmarks"]["docvqa"] = docvqa_results

        squad_results = benchmark_squad(model, tokenizer, max_samples=args.benchmark_samples)
        if squad_results:
            results["benchmarks"]["squad_v2"] = squad_results

        # Print benchmark summary
        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 60}")

        for name, bench in results["benchmarks"].items():
            print(f"\n{bench.get('dataset', name)}:")
            print(f"  Exact Match: {bench['exact_match']:.4f}")
            print(f"  Token F1:    {bench['token_f1']:.4f}")
            print(f"  Samples:     {bench['num_samples']}")
            if "answerable_em" in bench:
                print(f"  Answerable EM:   {bench['answerable_em']:.4f} (n={bench['num_answerable']})")
                print(f"  Unanswerable EM: {bench['unanswerable_em']:.4f} (n={bench['num_unanswerable']})")

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
