"""Run exploratory issue-type taxonomy analysis without creating training data."""

from dataset_inspection.taxonomy_analysis import analyze_taxonomy, print_taxonomy_summary, write_taxonomy_report


def main() -> None:
    """Analyze raw labels and write the taxonomy report."""
    summary = analyze_taxonomy()
    write_taxonomy_report(summary)
    print_taxonomy_summary(summary)


if __name__ == "__main__":
    main()
