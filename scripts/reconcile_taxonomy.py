"""Audit taxonomy counts against the normalized labeled dataset."""

from dataset_inspection.taxonomy_reconciliation import _reconcile, print_reconciliation_summary, write_reconciliation_report


def main() -> None:
    """Run the taxonomy reconciliation audit."""
    summary = _reconcile()
    write_reconciliation_report(summary)
    print_reconciliation_summary(summary)


if __name__ == "__main__":
    main()
