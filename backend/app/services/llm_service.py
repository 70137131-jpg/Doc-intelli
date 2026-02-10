import json
from typing import AsyncIterator

import google.generativeai as genai

from app.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)

DOCUMENT_TYPES = ["Invoice", "Contract", "Report", "Resume", "Letter", "Other"]

EXTRACTION_SCHEMAS = {
    "Invoice": {
        "vendor_name": "Name of the vendor/supplier",
        "invoice_number": "Invoice number or ID",
        "date": "Invoice date",
        "due_date": "Payment due date",
        "total_amount": "Total amount",
        "tax_amount": "Tax amount if present",
        "line_items": "List of line items with description and amount",
        "currency": "Currency used",
    },
    "Contract": {
        "parties": "Names of the contracting parties",
        "effective_date": "Start date of the contract",
        "termination_date": "End date or termination date",
        "key_terms": "Key terms and conditions",
        "governing_law": "Governing law/jurisdiction",
        "contract_value": "Total contract value if mentioned",
    },
    "Report": {
        "title": "Report title",
        "author": "Author name(s)",
        "date": "Report date",
        "summary": "Brief summary or abstract",
        "key_findings": "Key findings or conclusions",
        "organization": "Publishing organization",
    },
    "Resume": {
        "name": "Candidate full name",
        "email": "Email address",
        "phone": "Phone number",
        "education": "Education history",
        "experience": "Work experience",
        "skills": "Technical and other skills",
        "location": "Location/address",
    },
    "Letter": {
        "sender": "Sender name/organization",
        "recipient": "Recipient name/organization",
        "date": "Letter date",
        "subject": "Subject or regarding",
        "key_points": "Main points of the letter",
    },
}


class LLMService:
    def __init__(self):
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        try:
            model = self.model
            if system_instruction:
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    system_instruction=system_instruction,
                )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMError(f"Generation failed: {e}")

    def generate_json(self, prompt: str, system_instruction: str | None = None) -> dict:
        try:
            model = self.model
            if system_instruction:
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    system_instruction=system_instruction,
                )
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            text = response.text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise LLMError("Failed to parse JSON response from LLM")
        except Exception as e:
            logger.error(f"LLM JSON generation failed: {e}")
            raise LLMError(f"JSON generation failed: {e}")

    async def generate_stream_async(self, prompt: str, system_instruction: str | None = None) -> AsyncIterator[str]:
        try:
            model = self.model
            if system_instruction:
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    system_instruction=system_instruction,
                )
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise LLMError(f"Streaming generation failed: {e}")

    def classify_document(self, text: str) -> dict | None:
        categories = ", ".join(DOCUMENT_TYPES)
        prompt = f"""Classify the following document into one of these categories: {categories}

Analyze the document content and return a JSON object with:
- "category": the document type (must be one of: {categories})
- "confidence": confidence score between 0.0 and 1.0
- "reasoning": brief explanation of why this classification was chosen

Document text (first ~2000 tokens):
---
{text[:6000]}
---

Return only valid JSON."""

        try:
            result = self.generate_json(prompt)
            if "category" in result and "confidence" in result:
                # Validate category
                if result["category"] not in DOCUMENT_TYPES:
                    result["category"] = "Other"
                return result
            return None
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return None

    def extract_fields(self, text: str, document_type: str) -> dict | None:
        schema = EXTRACTION_SCHEMAS.get(document_type)
        if not schema:
            return None

        fields_desc = "\n".join(f'- "{k}": {v}' for k, v in schema.items())

        prompt = f"""Extract the following fields from this {document_type} document.

Fields to extract:
{fields_desc}

Document text:
---
{text[:6000]}
---

Return a JSON object with the field names as keys and extracted values.
For fields that are lists (like line_items, education, experience, skills, key_terms, key_findings, key_points), return them as arrays of strings.
If a field cannot be found, set its value to null.
Return only valid JSON."""

        try:
            return self.generate_json(prompt)
        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return None

    def expand_query(self, query: str) -> list[str]:
        prompt = f"""Generate 2 alternative phrasings of this search query that might match relevant documents.
Keep the same intent but use different words.

Original query: "{query}"

Return a JSON array of 2 alternative query strings.
Example: ["alternative phrasing 1", "alternative phrasing 2"]"""

        try:
            result = self.generate_json(prompt)
            if isinstance(result, list):
                return result[:2]
            return []
        except Exception:
            return []

    def check_faithfulness(self, context: str, answer: str) -> dict:
        prompt = f"""Given the following context and answer, determine if the answer is faithful to the context.

Context:
---
{context[:4000]}
---

Answer:
---
{answer}
---

Return a JSON object with:
- "faithful": boolean, true if the answer is supported by the context
- "unsupported_claims": array of strings listing any claims not supported by the context
- "confidence": string, one of "high", "medium", "low"

Return only valid JSON."""

        try:
            return self.generate_json(prompt)
        except Exception:
            return {"faithful": True, "unsupported_claims": [], "confidence": "low"}
