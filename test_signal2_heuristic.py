import os

from dotenv import load_dotenv

from signals import groq_style_assessment, heuristic_perplexity_signal


def main() -> None:
    load_dotenv()

    samples = [
        "I went to the store after work and realized I'd forgotten my wallet, so I had to drive back home. It was annoying, but I laughed at myself on the way.",
        "In conclusion, leveraging synergistic paradigms enables stakeholders to optimize end-to-end outcomes while maintaining robust alignment across cross-functional objectives.",
        "Hello. This is short.",
    ]

    have_groq = bool(os.getenv("GROQ_API_KEY"))

    for idx, text in enumerate(samples, start=1):
        print(f"\n=== Sample {idx} ===")
        if have_groq:
            s1 = groq_style_assessment(text)
            print({"signal_1_style_score": s1.get("style_score")})
        else:
            print("Missing GROQ_API_KEY; skipping signal_1 Groq call")

        s2 = heuristic_perplexity_signal(text)
        print({"signal_2_perplexity_score": s2.get("perplexity_score"), "metrics": s2.get("metrics")})


if __name__ == "__main__":
    main()
