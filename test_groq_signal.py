import os

from dotenv import load_dotenv

from signals import groq_style_assessment


def main() -> None:
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print("Missing GROQ_API_KEY; skipping Groq signal test")
        return

    samples = [
        "I went to the store after work and realized I'd forgotten my wallet, so I had to drive back home. It was annoying, but I laughed at myself on the way.",
        "In conclusion, leveraging synergistic paradigms enables stakeholders to optimize end-to-end outcomes while maintaining robust alignment across cross-functional objectives.",
        "Hello. This is short.",
    ]

    for idx, text in enumerate(samples, start=1):
        result = groq_style_assessment(text)
        print(f"\nSample {idx} result:\n{result}")


if __name__ == "__main__":
    main()
