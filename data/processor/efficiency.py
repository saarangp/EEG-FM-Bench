"""
Stratified subsampling utility for data efficiency evaluation.
"""

import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def stratified_subsample(
    dataset: Dataset,
    fraction: float,
    seed: int
) -> Dataset:
    """
    Perform stratified subsampling of a dataset by label column.

    Args:
        dataset: HuggingFace Dataset with 'label' column
        fraction: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Subsampled dataset maintaining class distribution
    """
    if fraction >= 1.0:
        return dataset

    if 'label' not in dataset.column_names:
        raise ValueError("Dataset must have 'label' column for stratified sampling")

    n_original = len(dataset)
    indices = np.arange(n_original)
    labels = np.array(dataset['label'])

    # Check for minimum samples per class
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_samples_needed = max(2, int(np.ceil(1.0 / fraction)))

    for label, count in zip(unique_labels, label_counts):
        if count < min_samples_needed:
            logger.warning(
                f"Class {label} has only {count} samples, may cause issues with "
                f"stratified split at {fraction*100:.0f}% fraction"
            )

    try:
        # Use train_test_split to get stratified subsample
        # We discard the "test" portion and keep only the "train" portion
        kept_indices, _ = train_test_split(
            indices,
            train_size=fraction,
            stratify=labels,
            random_state=seed
        )

        subsampled = dataset.select(sorted(kept_indices.tolist()))
        logger.info(f"Subsampled: {n_original} -> {len(subsampled)} ({fraction*100:.0f}%)")

        return subsampled

    except ValueError as e:
        # Fall back to random sampling if stratified fails (e.g., too few samples)
        logger.warning(f"Stratified sampling failed: {e}. Falling back to random sampling.")
        rng = np.random.default_rng(seed)
        n_samples = int(n_original * fraction)
        kept_indices = rng.choice(indices, size=n_samples, replace=False)
        subsampled = dataset.select(sorted(kept_indices.tolist()))
        logger.info(f"Random subsampled: {n_original} -> {len(subsampled)} ({fraction*100:.0f}%)")
        return subsampled
