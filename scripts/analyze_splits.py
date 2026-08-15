"""Construct normalized labels and analyze split strategies without writing splits."""

from dataset_inspection.split_analysis import analyze_splits, print_split_analysis_summary, write_split_analysis_report


def main() -> None:
    """Construct the normalized dataset and write split-analysis metadata."""
    summary = analyze_splits()
    write_split_analysis_report(summary)
    print_split_analysis_summary(summary)


if __name__ == "__main__":
    main()
