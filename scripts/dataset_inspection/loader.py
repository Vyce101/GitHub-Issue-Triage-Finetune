"""Dataset loading for raw GitHub issue inspection."""

from datasets import DatasetDict, load_dataset

from .config import DATASET_ID


def load_dataset_splits() -> DatasetDict:
    """Load the named dataset without selecting, filtering, or splitting rows."""
    loaded = load_dataset(DATASET_ID)
    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({"default": loaded})
