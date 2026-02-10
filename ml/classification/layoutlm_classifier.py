"""
LayoutLMv3 for layout-aware document classification and token classification.

Uses both text AND visual layout features for:
1. Document type classification (whole-document)
2. Token classification for field extraction (per-token labels)

Requires: pip install transformers[torch] Pillow
Optional GPU: pip install torch torchvision (CUDA)
"""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)

DOCUMENT_LABELS = ["Invoice", "Contract", "Report", "Resume", "Letter", "Other"]
FIELD_LABELS = ["O", "B-VENDOR", "I-VENDOR", "B-DATE", "I-DATE", "B-AMOUNT", "I-AMOUNT",
                "B-NAME", "I-NAME", "B-EMAIL", "I-EMAIL", "B-PHONE", "I-PHONE",
                "B-ADDRESS", "I-ADDRESS"]


class LayoutLMv3Classifier:
    """Layout-aware document classification using LayoutLMv3."""

    MODEL_NAME = "microsoft/layoutlmv3-base"

    def __init__(self, model_dir: str | None = None, task: str = "classification"):
        self.task = task
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME, apply_ocr=False)

        if task == "classification":
            labels = DOCUMENT_LABELS
            if model_dir and os.path.exists(model_dir):
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    model_dir, num_labels=len(labels)
                )
            else:
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    self.MODEL_NAME, num_labels=len(labels)
                )
            self.labels = labels
        else:
            labels = FIELD_LABELS
            if model_dir and os.path.exists(model_dir):
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    model_dir, num_labels=len(labels)
                )
            else:
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.MODEL_NAME, num_labels=len(labels)
                )
            self.labels = labels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_document_type(
        self,
        words: list[str],
        boxes: list[list[int]],
        image: Image.Image,
    ) -> dict:
        """Classify a document given OCR words, bounding boxes, and page image.

        Args:
            words: List of OCR-detected words
            boxes: List of [x0, y0, x1, y1] bounding boxes (normalized 0-1000)
            image: PIL Image of the document page

        Returns:
            {"category": str, "confidence": float, "scores": dict}
        """
        encoding = self.processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True, max_length=512, padding="max_length",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_idx = probs.argmax().item()

        return {
            "category": self.labels[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "scores": {label: float(probs[i]) for i, label in enumerate(self.labels)},
            "method": "layoutlmv3",
        }

    def predict_fields(
        self,
        words: list[str],
        boxes: list[list[int]],
        image: Image.Image,
    ) -> list[dict]:
        """Token classification to extract fields with their positions.

        Returns list of {"word": str, "label": str, "confidence": float}
        """
        encoding = self.processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True, max_length=512, padding="max_length",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        predictions = probs.argmax(dim=-1)

        results = []
        for i, (word, pred_idx) in enumerate(zip(words, predictions[:len(words)])):
            label = self.labels[pred_idx]
            if label != "O":
                results.append({
                    "word": word,
                    "label": label,
                    "confidence": float(probs[i][pred_idx]),
                    "box": boxes[i] if i < len(boxes) else None,
                })

        # Group B/I tokens into fields
        return self._group_bio_tokens(results)

    @staticmethod
    def _group_bio_tokens(tokens: list[dict]) -> list[dict]:
        """Group BIO-tagged tokens into complete field entities."""
        entities = []
        current = None

        for token in tokens:
            label = token["label"]
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                field_type = label[2:]
                current = {
                    "field_type": field_type,
                    "value": token["word"],
                    "confidence": token["confidence"],
                }
            elif label.startswith("I-") and current:
                field_type = label[2:]
                if field_type == current["field_type"]:
                    current["value"] += " " + token["word"]
                    current["confidence"] = min(current["confidence"], token["confidence"])

        if current:
            entities.append(current)

        return entities

    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./output/layoutlmv3",
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
    ):
        """Fine-tune LayoutLMv3 on labeled document data."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        return trainer.evaluate()
