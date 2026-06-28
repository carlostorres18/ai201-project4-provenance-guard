import os

from dotenv import load_dotenv

from signals import (
    classify_confidence,
    compute_confidence,
    groq_style_assessment,
    heuristic_perplexity_signal,
    label_for_classification,
)


def run_case(name: str, text: str) -> None:
    s1 = groq_style_assessment(text)
    s2 = heuristic_perplexity_signal(text)

    style_score = float(s1["style_score"])
    perplexity_score = float(s2["perplexity_score"])

    confidence = compute_confidence(style_score=style_score, perplexity_score=perplexity_score)
    classification = classify_confidence(confidence)
    label = label_for_classification(classification)

    print(f"\n=== {name} ===")
    print({
        "style_score": style_score,
        "perplexity_score": perplexity_score,
        "confidence": confidence,
        "classification": classification,
        "label": label,
        "signal_2_reason": s2.get("reasoning"),
        "signal_2_metrics": s2.get("metrics"),
    })


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Missing GROQ_API_KEY")

    ai_text = (
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment."
    )

    human_text = (
        "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it and "
        "i was thirsty for like three hours after. my friend got the spicy version and "
        "said it was better. probably won't go back unless someone drags me there"
    )

    borderline_formal_human = (
        "The relationship between monetary policy and asset price inflation has been "
        "extensively studied in the literature. Central banks face a fundamental tension "
        "between their mandate for price stability and the unintended consequences of "
        "prolonged low interest rates on equity and real estate valuations."
    )

    borderline_edited_ai = (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
        "flexibility and no commute on one side, isolation and blurred work-life boundaries "
        "on the other. Studies show productivity varies widely by individual and role type."
    )

    run_case("Clearly AI-generated", ai_text)
    run_case("Clearly human-written", human_text)
    run_case("Borderline formal human", borderline_formal_human)
    run_case("Borderline lightly edited AI", borderline_edited_ai)


if __name__ == "__main__":
    main()
