import json
import os
import math
import re
from typing import Any

from groq import Groq


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def groq_style_assessment(text: str) -> dict[str, Any]:
    client = Groq()
    model = os.getenv("GROQ_STYLE_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are scoring AI-likeness. Higher means more AI-generated. "
                    "Return ONLY a single JSON object with keys: "
                    "style_score (number 0.0-1.0 where 0.0=strongly human-written and 1.0=strongly AI-generated), "
                    "reasoning (string). "
                    "Calibration examples: "
                    "Example A (casual, messy, personal rant) should be ~0.05-0.25. "
                    "Example B (generic academic tone, hedging like 'it is important to note', 'furthermore') should be ~0.75-0.95. "
                    "Example C (formal but specific, content-rich academic prose with concrete claims) should be ~0.35-0.60. "
                    "Important: do not treat formality alone as AI. Score higher when the writing is generic, templated, or padded with hedging; "
                    "score lower when the writing is specific, grounded, and shows nuanced tradeoffs."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Example C text: The relationship between monetary policy and asset price inflation has been extensively studied in the literature. "
                    "Central banks face a fundamental tension between their mandate for price stability and the unintended consequences of prolonged low "
                    "interest rates on equity and real estate valuations."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "style_score": 0.5,
                        "reasoning": "Formal but specific and content-rich; do not over-penalize for academic tone alone.",
                    }
                ),
            },
            {
                "role": "user",
                "content": (
                    "Example D text: I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
                    "flexibility and no commute on one side, isolation and blurred work-life boundaries on the other. "
                    "Studies show productivity varies widely by individual and role type."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "style_score": 0.65,
                        "reasoning": "Polished, balanced, and generic-sounding; could be lightly edited AI, so mid-high AI-likeness.",
                    }
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


def groq_perplexity_assessment(text: str) -> dict[str, Any]:
    client = Groq()
    model = os.getenv("GROQ_PERPLEXITY_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Estimate how statistically predictable the submitted text is, "
                    "as a proxy for perplexity analysis. "
                    "Return ONLY a single JSON object with keys: "
                    "perplexity_score (number 0.0-1.0 where 0.0=strongly human-like and 1.0=strongly AI-like), "
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

    perplexity_score = payload.get("perplexity_score")
    if not isinstance(perplexity_score, (int, float)):
        raise ValueError("Groq payload missing numeric perplexity_score")

    payload["perplexity_score"] = clamp01(float(perplexity_score))
    return payload


def compute_confidence(style_score: float, perplexity_score: float) -> float:
    return clamp01((clamp01(style_score) + clamp01(perplexity_score)) / 2.0)


def classify_confidence(confidence: float) -> str:
    confidence = clamp01(confidence)
    if confidence <= 0.39:
        return "likely_human"
    if confidence >= 0.61:
        return "likely_ai"
    return "uncertain"


def label_for_classification(classification: str) -> str:
    if classification == "likely_ai":
        return "AI-Generated (High Confidence)"
    if classification == "likely_human":
        return "Human-Written (High Confidence)"
    return "Attribution Uncertain"


def heuristic_perplexity_signal(text: str) -> dict[str, Any]:
    normalized_text = re.sub(r"\s+", " ", text.strip())
    if not normalized_text:
        return {"perplexity_score": 0.5, "reasoning": "empty_text"}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized_text) if s.strip()]
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", normalized_text)]

    if len(tokens) < 30 or len(sentences) < 2:
        return {"perplexity_score": 0.5, "reasoning": "too_short"}

    token_freq: dict[str, int] = {}
    for token in tokens:
        token_freq[token] = token_freq.get(token, 0) + 1

    total = len(tokens)
    entropy = 0.0
    for count in token_freq.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)

    entropy_ai = clamp01((4.1 - entropy) / 0.8)

    bigrams = list(zip(tokens, tokens[1:]))
    bigram_freq: dict[tuple[str, str], int] = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

    repeated_bigrams = sum(count for count in bigram_freq.values() if count >= 2)
    bigram_repeat_rate = repeated_bigrams / max(1, len(bigrams))
    repetition_ai = clamp01((bigram_repeat_rate - 0.05) / (0.25 - 0.05))

    sentence_lengths: list[int] = []
    for sentence in sentences:
        sentence_tokens = re.findall(r"[A-Za-z']+", sentence)
        if sentence_tokens:
            sentence_lengths.append(len(sentence_tokens))

    if not sentence_lengths:
        return {"perplexity_score": 0.5, "reasoning": "no_sentences"}

    mean_len = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((length - mean_len) ** 2 for length in sentence_lengths) / len(sentence_lengths)
    std_len = variance**0.5
    burstiness_ai = clamp01((10.0 - std_len) / 10.0)

    perplexity_score = clamp01((entropy_ai + repetition_ai + burstiness_ai) / 3.0)
    return {
        "perplexity_score": perplexity_score,
        "metrics": {
            "token_entropy": entropy,
            "bigram_repeat_rate": bigram_repeat_rate,
            "sentence_length_std": std_len,
        },
        "reasoning": "heuristic_proxy",
    }
