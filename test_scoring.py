from signals import classify_confidence, compute_confidence


def main() -> None:
    assert classify_confidence(0.0) == "likely_human"
    assert classify_confidence(0.39) == "likely_human"
    assert classify_confidence(0.40) == "uncertain"
    assert classify_confidence(0.60) == "uncertain"
    assert classify_confidence(0.61) == "likely_ai"
    assert classify_confidence(1.0) == "likely_ai"

    assert compute_confidence(0.0, 0.0) == 0.0
    assert compute_confidence(1.0, 1.0) == 1.0
    assert compute_confidence(0.2, 0.8) == 0.5

    print("scoring ok")


if __name__ == "__main__":
    main()
