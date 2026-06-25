from intent_classifier import MODEL_PATH, benchmark, benchmark_summary, get_model_stats, train_if_needed


def main():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    train_if_needed()
    print("Model stats:", get_model_stats())
    print("Benchmark summary:", benchmark_summary())
    print(benchmark().to_string(index=False))


if __name__ == "__main__":
    main()
