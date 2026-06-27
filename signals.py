import json
import os
from typing import Any

from groq import Groq


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def groq_style_assessment(text: str) -> dict[str, Any]:
    client = Groq()
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Assess whether the submitted text is AI-generated vs human-written. "
                    "Return ONLY a single JSON object with keys: "
                    "style_score (number 0.0-1.0 where 0.0=strongly human-like and 1.0=strongly AI-like), "
                    "reasoning (string)."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned empty content")

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Groq returned non-object JSON")

    style_score = payload.get("style_score")
    if not isinstance(style_score, (int, float)):
        raise ValueError("Groq payload missing numeric style_score")

    payload["style_score"] = clamp01(float(style_score))
    return payload
