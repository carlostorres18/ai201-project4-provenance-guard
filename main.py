import os
import re
import uuid
from typing import Any
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from audit import append_audit_log, find_first_event, get_audit_log_entries, update_first_event
from labels import generate_transparency_label
from signals import (
    classify_confidence,
    compute_confidence,
    heuristic_perplexity_signal,
    groq_style_assessment,
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_linguistic_style_score(text: str) -> float:
    normalized_text = re.sub(r"\s+", " ", text.strip())
    if not normalized_text:
        return 0.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized_text) if s.strip()]
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", normalized_text)]

    if len(words) < 30 or len(sentences) < 2:
        return 0.5

    sentence_lengths = []
    for sentence in sentences:
        sentence_words = re.findall(r"[A-Za-z']+", sentence)
        if sentence_words:
            sentence_lengths.append(len(sentence_words))

    if not sentence_lengths:
        return 0.5

    mean_sentence_len = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((length - mean_sentence_len) ** 2 for length in sentence_lengths) / len(
        sentence_lengths
    )
    std_sentence_len = variance**0.5

    unique_words = set(words)
    type_token_ratio = safe_div(len(unique_words), len(words))

    total_words = len(words)
    word_freq: dict[str, int] = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    repeated_tokens = sum(count for count in word_freq.values() if count >= 3)
    repetition_ratio = safe_div(repeated_tokens, total_words)

    syllables = 0
    for word in words:
        groups = re.findall(r"[aeiouy]+", word)
        syllables += max(1, len(groups))

    words_per_sentence = safe_div(total_words, len(sentence_lengths))
    syllables_per_word = safe_div(syllables, total_words)

    flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    readability_ai = clamp01((80.0 - flesch_reading_ease) / 80.0)

    uniformity_ai = clamp01((18.0 - std_sentence_len) / 18.0)
    diversity_ai = clamp01((0.55 - type_token_ratio) / 0.55)
    repetition_ai = clamp01((repetition_ratio - 0.12) / (0.45 - 0.12))

    score = (uniformity_ai + diversity_ai + repetition_ai + readability_ai) / 4.0
    return clamp01(score)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")




def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
    )

    @app.get("/log")
    @limiter.limit(os.getenv("LOG_RATE_LIMIT", "30 per minute"))
    def log() -> tuple[Any, int]:
        limit_raw = request.args.get("limit", "50")
        try:
            limit = int(limit_raw)
        except ValueError:
            return jsonify({"error": "Query param 'limit' must be an integer"}), 400

        return jsonify({"entries": get_audit_log_entries(limit=limit)}), 200

    @app.post("/submit")
    @limiter.limit(os.getenv("SUBMIT_RATE_LIMIT", "10 per minute;100 per day"))
    def submit() -> tuple[Any, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be JSON"}), 400

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Field 'text' is required"}), 400

        creator_id = payload.get("creator_id")
        if not isinstance(creator_id, str) or not creator_id.strip():
            return jsonify({"error": "Field 'creator_id' is required"}), 400

        content_id = str(uuid.uuid4())
        timestamp = utc_timestamp()

        try:
            if os.getenv("MOCK_SIGNALS", "0") == "1":
                signal_1 = {"style_score": 0.5, "reasoning": "mock"}
                signal_2 = {"perplexity_score": 0.5, "reasoning": "mock"}
            else:
                signal_1 = groq_style_assessment(text)
                signal_2 = heuristic_perplexity_signal(text)
        except Exception as exc:
            append_audit_log(
                {
                    "timestamp": timestamp,
                    "event": "submit_error",
                    "content_id": content_id,
                    "creator_id": creator_id,
                    "error": str(exc),
                }
            )
            return (
                jsonify(
                    {
                        "content_id": content_id,
                        "error": "signal_failed",
                        "message": str(exc),
                    }
                ),
                502,
            )

        style_score = float(signal_1["style_score"])
        perplexity_score = float(signal_2["perplexity_score"])

        confidence = compute_confidence(style_score=style_score, perplexity_score=perplexity_score)
        attribution = classify_confidence(confidence)
        transparency_label = generate_transparency_label(confidence)

        response = {
            "content_id": content_id,
            "creator_id": creator_id,
            "signal_1": signal_1,
            "signal_2": signal_2,
            "attribution": attribution,
            "confidence": confidence,
            "label": {"title": transparency_label.title, "message": transparency_label.message},
        }

        append_audit_log(
            {
                "timestamp": timestamp,
                "event": "submit",
                "content_id": content_id,
                "creator_id": creator_id,
                "attribution": attribution,
                "confidence": confidence,
                "label": {"title": transparency_label.title, "message": transparency_label.message},
                "style_score": style_score,
                "perplexity_score": perplexity_score,
                "status": "classified",
            }
        )

        return jsonify(response), 200

    @app.post("/appeal")
    @limiter.limit(os.getenv("APPEAL_RATE_LIMIT", "10 per hour"))
    def appeal() -> tuple[Any, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be JSON"}), 400

        content_id = payload.get("content_id")
        if not isinstance(content_id, str) or not content_id.strip():
            return jsonify({"error": "Field 'content_id' is required"}), 400

        creator_reasoning = payload.get("creator_reasoning")
        if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
            return jsonify({"error": "Field 'creator_reasoning' is required"}), 400

        timestamp = utc_timestamp()
        submission = find_first_event(content_id=content_id, event="submit")
        if submission is None:
            return jsonify({"error": "content_id not found"}), 404

        updated = update_first_event(content_id=content_id, event="submit", updates={"status": "under_review"})
        if not updated:
            return jsonify({"error": "failed to update status"}), 500

        appeal_id = str(uuid.uuid4())
        append_audit_log(
            {
                "timestamp": timestamp,
                "event": "appeal",
                "appeal_id": appeal_id,
                "content_id": content_id,
                "creator_id": submission.get("creator_id"),
                "creator_reasoning": creator_reasoning,
                "original_attribution": submission.get("attribution"),
                "original_confidence": submission.get("confidence"),
                "original_style_score": submission.get("style_score"),
                "original_perplexity_score": submission.get("perplexity_score"),
                "status": "under_review",
            }
        )

        return (
            jsonify(
                {
                    "content_id": content_id,
                    "appeal_id": appeal_id,
                    "status": "under_review",
                    "message": "appeal received",
                }
            ),
            200,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), debug=True)
