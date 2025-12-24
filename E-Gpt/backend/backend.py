import os
import re
import json
import random
import requests
import fitz  # PyMuPDF
from dataclasses import dataclass
from json_repair import repair_json


# =====================================================================
# 🧠 MODEL CONFIG
# =====================================================================
@dataclass
class LlamaConfig:
    model_id: str = "llama3:8b"
    max_new_tokens: int = 1500
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05


# =====================================================================
# 📄 PDF TEXT EXTRACTION
# =====================================================================
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])

        # Strip References
        text = re.split(r"\n\s*references\s*\n", text, flags=re.IGNORECASE)[0]

        # Remove ALL CAPS headings
        text = "\n".join(
            [line for line in text.splitlines()
             if not re.match(r"^[A-Z\s]{4,}$", line.strip())]
        )

        # Remove repeated page headers
        lines = text.splitlines()
        seen = {}
        for line in lines:
            t = line.strip()
            if len(t.split()) <= 4:
                seen[t] = seen.get(t, 0) + 1
        repeated = {x for x in seen if seen[x] > 3}
        text = "\n".join([l for l in lines if l.strip() not in repeated])

        return text.strip()

    except Exception as e:
        print("⚠️ PDF extraction error:", e)
        return ""

def extract_text_from_multiple_pdfs(files):
    combined = ""
    for f in files:
        try:
            f.seek(0)
            combined += extract_text_from_pdf(f) + "\n\n"
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
    return combined.strip()

# =====================================================================
# ⚙️ JSON REPAIR — BULLET-PROOF VERSION
# =====================================================================
def safe_json_extract(text: str):
    """
    Repairs malformed JSON produced by LLaMA / Mistral / Phi models.
    Handles:
      - Missing quotes
      - Unquoted A/B/C/D keys
      - Trailing commas
      - Broken brackets
      - Extra text around JSON
    """
    if not text:
        return []

    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Extract only JSON array region
    start = text.find('[')
    end = text.rfind(']')
    candidate = text[start:end + 1] if start != -1 and end != -1 else text

    # -------------------------------------------------------------
    # 🔥 CRITICAL FIXES — ensure JSON is structurally valid
    # -------------------------------------------------------------

    # 1) Quote keys A, B, C, D correctly
    candidate = re.sub(r'(?<!")\b([A-D])\b(?=\s*:)', r'"\1"', candidate)

    # 2) Quote dictionary keys missing quotes
    candidate = re.sub(r'(?m)^\s*([a-zA-Z_]+)\s*:', r'"\1":', candidate)

    # 3) Remove trailing commas before } or ]
    candidate = re.sub(r',\s*(?=[}\]])', '', candidate)

    # 4) Remove strange trailing garbage
    candidate = re.sub(r'[\x00-\x1F]+$', '', candidate)

    # 5) Fix unescaped quotes inside text
    candidate = candidate.replace('\\"', '"')

    # Try repair_json
    try:
        repaired = repair_json(candidate)
        return json.loads(repaired)
    except Exception as e:
        print("❌ JSON final repair failed:", e)
        return []


# =====================================================================
# 🚀 QUIZ GENERATOR — Stable, Full Version
# =====================================================================
def generate_quiz_from_text(content: str, cfg: LlamaConfig, num_questions=10, seed=42):

    random.seed(seed)

    prompt = f"""
You are an expert educator. Create exactly **{num_questions}** high-quality MCQs.

REQUIREMENTS:
- Output ONLY a **JSON array**
- No text before or after
- Keys MUST be quoted
- Options MUST be inside: "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}

CORRECT FORMAT:
[
  {{
    "question": "text",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_answer": "A",
    "explanation": "text"
  }}
]

MATERIAL:
{content[:2500]}
"""

    raw_text = ""
    metadata = {"num_questions": num_questions, "model": cfg.model_id}

    try:
        print("🚀 Calling Ollama model:", cfg.model_id)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": cfg.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": cfg.temperature,
                    "num_predict": cfg.max_new_tokens,
                    "top_p": cfg.top_p,
                    "repeat_penalty": cfg.repetition_penalty,
                },
            },
            timeout=180
        )

        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "").strip()

        print("🔍 RAW MODEL OUTPUT (first 300 chars):")
        print(raw_text[:300])

        # CLEAN MODEL OUTPUT
        if raw_text.startswith("Here are"):
            raw_text = raw_text.split("\n", 1)[-1].strip()

        parsed = safe_json_extract(raw_text)

        # VALIDATE & CLEAN RESULTS
        final = []
        for q in parsed:
            if isinstance(q, dict) and "question" in q and "options" in q and "correct_answer" in q:
                final.append(q)

        # Trim
        final = final[:num_questions]

        # Fill missing
        while len(final) < num_questions:
            final.append({
                "question": "⚠️ Question generation failed.",
                "options": {"A": "—", "B": "—", "C": "—", "D": "—"},
                "correct_answer": "A",
                "explanation": "Model failed to generate this item."
            })

        print(f"✅ Final quiz length: {len(final)}")

        return final, raw_text, metadata

    except Exception as e:
        print("❌ MODEL ERROR:", e)
        return [], raw_text, metadata
