import numpy as np
import pytest
from helper_functions import clean_curves, clean_curves_together, filter_by_group_definitions, get_all_pairs


def test_clean_curves_removes_nan_and_short_curves():
    curves = [np.array([1, 2, 3, np.nan]), np.array([1]), np.array([1, 1, 1])]
    labels = [0, 1, 2]
    cleaned, cleaned_labels = clean_curves(curves, labels, min_length=2)
    print(cleaned)
    assert len(cleaned) == 2
    assert all(~np.isnan(c).any() for c in cleaned)
    assert cleaned_labels.tolist() == [0, 2]


def test_clean_curves_flat_filtering():
    curves = [np.ones(100), np.linspace(0, 1, 100)]
    labels = [0, 1]
    cleaned, cleaned_labels = clean_curves(curves, labels, remove_flat=True)
    assert len(cleaned) == 1
    assert cleaned_labels[0] == 1


def test_clean_curves_together_all_curves_must_be_valid():
    tr = [np.ones(100), np.linspace(0, 1, 100)]
    va = [np.ones(100), np.linspace(0, 1, 100)]
    te = [np.ones(100), np.linspace(0, 1, 100)]
    labels = [0, 1]

    cleaned = clean_curves_together(tr, va, te, labels, remove_flat=True)

    # only the second set should remain
    tr_clean, va_clean, te_clean, labels_clean = cleaned
    assert len(tr_clean) == len(va_clean) == len(te_clean) == len(labels_clean) == 1
    assert labels_clean[0] == 1


def test_filter_by_group_definitions_curves_present():
    curves = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    original_labels = [10, 20, 30]
    group_defs = {'A': ['Alice'], 'B': ['Bob']}
    zoo = {10: 'Alice', 20: 'Charlie', 30: 'Bob'}

    filtered_curves, filtered_labels, group_labels = filter_by_group_definitions(curves, original_labels, group_defs,
                                                                                 zoo)

    assert len(filtered_curves) == 2
    assert set(group_labels) == {'A', 'B'}


def test_clean_curves_all_nan():
    curves = [np.array([np.nan, np.nan, np.nan])]
    labels = [0]
    cleaned, cleaned_labels = clean_curves(curves, labels)
    assert len(cleaned) == 0
    assert cleaned_labels.size == 0


def test_clean_curves_included_labels():
    curves = [np.arange(100), np.arange(100)]
    labels = [1, 2]
    cleaned, cleaned_labels = clean_curves(curves, labels, included_labels=[1])
    assert len(cleaned) == 1
    assert cleaned_labels[0] == 1


def test_clean_curves_together_nan_in_test_set():
    tr = [np.arange(100)]
    va = [np.arange(100)]
    te = [np.array([1, 2, 4, 5, np.nan])]
    labels = [0]

    tr_clean, va_clean, te_clean, labels_clean = clean_curves_together(tr, va, te, labels, min_length=5)
    assert len(tr_clean) == len(va_clean) == len(te_clean) == 0


def test_filter_by_group_definitions_no_match():
    curves = [np.arange(10)]
    original_labels = [99]
    group_defs = {'A': ['Alice']}
    zoo = {10: 'Bob'}

    filtered_curves, filtered_labels, group_labels = filter_by_group_definitions(curves, original_labels, group_defs,
                                                                                 zoo)
    assert len(filtered_curves) == 0
    assert filtered_labels.size == 0


def test_filter_by_group_definitions_no_curves():
    curves = None
    original_labels = [1, 2]
    group_defs = {'X': ['a'], 'Y': ['b']}
    zoo = {1: 'a', 2: 'c'}

    filtered_curves, filtered_labels, group_labels = filter_by_group_definitions(curves, original_labels, group_defs, zoo)
    assert filtered_curves == []
    assert filtered_labels.tolist() == [1]
    assert group_labels.tolist() == ['X']


