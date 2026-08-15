"""Run the dataset inspection workflow from its command-line entry point."""

from .inspection import inspect_dataset
from .report import print_human_summary, write_report


def main() -> None:
    """Run the raw dataset inspection and write its JSON report."""
    summary = inspect_dataset()
    write_report(summary)
    print_human_summary(summary)
