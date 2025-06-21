import numpy as np
from itertools import combinations


def clean_curves(curves, labels, min_length=60, included_labels=None, remove_flat=False, flat_std_threshold=1e-3,
                 flat_range_threshold=1e-2):
    cleaned_curves = []
    cleaned_labels = []

    for curve, label in zip(curves, labels):
        if included_labels is not None and label not in included_labels:
            continue

        # remove NaN values
        cleaned_curve = curve[~np.isnan(curve)]

        # skip if too short
        if cleaned_curve.size < min_length:
            continue

        std_dev = np.std(cleaned_curve)
        val_range = np.max(cleaned_curve) - np.min(cleaned_curve)

        if remove_flat and (std_dev < flat_std_threshold or val_range < flat_range_threshold):
            continue  # skip flat curves

        # keep the curve
        cleaned_curves.append(cleaned_curve)
        cleaned_labels.append(label)

    return cleaned_curves, np.array(cleaned_labels)


def clean_curves_together(train_curves, valid_curves,
                          test_curves, labels,
                          min_length=60, included_labels=None,
                          remove_flat=False, flat_std_threshold=1e-3,
                          flat_range_threshold=1e-2):
    cleaned_train_curves = []

    cleaned_valid_curves = []

    cleaned_test_curves = []
    cleaned_labels = []

    for tr_curve, va_curve, te_curve, label in zip(
            train_curves, valid_curves, test_curves, labels):

        if included_labels is not None and label not in included_labels:
            continue

        def is_valid(curve):
            curve = curve[~np.isnan(curve)]
            if curve.size < min_length:
                return False
            if remove_flat:
                std_dev = np.std(curve)
                val_range = np.max(curve) - np.min(curve)
                if std_dev < flat_std_threshold or val_range < flat_range_threshold:
                    return False
            return True

        tr_clean = tr_curve[~np.isnan(tr_curve)]
        va_clean = va_curve[~np.isnan(va_curve)]
        te_clean = te_curve[~np.isnan(te_curve)]

        if not (is_valid(tr_clean) and is_valid(va_clean) and is_valid(te_clean)):
            continue

        cleaned_train_curves.append(tr_clean)

        cleaned_valid_curves.append(va_clean)

        cleaned_test_curves.append(te_clean)

        cleaned_labels.append(label)

    return (cleaned_train_curves, cleaned_valid_curves,
            cleaned_test_curves, np.array(cleaned_labels))


def filter_by_group_definitions(curves, original_labels, group_definitions, learner_zoo):
    # build inverse map: learner index -> group
    learner_to_group = {}
    for group, learners in group_definitions.items():
        for name in learners:
            if name in learner_zoo.values():
                idx = list(learner_zoo.keys())[list(learner_zoo.values()).index(name)]
                learner_to_group[idx] = group

    # filter curves and assign group labels
    filtered_curves = []
    filtered_labels = []
    group_labels = []

    for curve, original_label in zip(curves, original_labels):
        if original_label in learner_to_group:
            filtered_curves.append(curve)
            filtered_labels.append(original_label)
            group_labels.append(learner_to_group[original_label])

    return filtered_curves, np.array(filtered_labels), np.array(group_labels)


def filter_by_group_definitions_together(train_curves, valid_curves, test_curves,
                                         original_labels, group_definitions, learner_zoo):

    # build inverse map: learner index -> group name
    learner_to_group = {}
    for group, learners in group_definitions.items():
        for name in learners:
            if name in learner_zoo.values():
                idx = list(learner_zoo.keys())[list(learner_zoo.values()).index(name)]
                learner_to_group[idx] = group

    # filter each curve triple
    filtered_train_curves = []
    filtered_valid_curves = []
    filtered_test_curves = []
    filtered_labels = []
    group_labels = []

    for tr_curve, va_curve, te_curve, original_label in zip(train_curves, valid_curves, test_curves, original_labels):
        if original_label in learner_to_group:
            filtered_train_curves.append(tr_curve)
            filtered_valid_curves.append(va_curve)
            filtered_test_curves.append(te_curve)
            filtered_labels.append(original_label)
            group_labels.append(learner_to_group[original_label])

    return (filtered_train_curves, filtered_valid_curves, filtered_test_curves,
            np.array(filtered_labels), np.array(group_labels))


def get_all_pairs(n=24, r=2):
    return list(combinations(range(n), r))
