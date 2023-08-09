from typing import Tuple, List, Set
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from htm_source.utils.general import make_windows

EPS = 1e-12
Slice = namedtuple('Slice', ['start', 'end'])


def calc_conf_mat(preds: Set[Slice], targets: Set[Slice], data_len: int) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix values for given predictions and labels.
    Both predictions and labels (targets) are supplied as a set of slices, representing windows.

    returns: tp, fp, fn, tn
    """

    tp = 0
    hits = set()
    was_hit = set()

    for tag in targets:
        # for each label window, mark all related (intersecting) predictions as hits (counts as 1 hit per tag)
        if related := set(filter(lambda x: (tag.start <= x.start < tag.end) or (tag.start <= x.end < tag.end), preds)):
            hits.update(related)
            was_hit.add(tag)
            tp += 1

    # fp are all predictions that didn't hit at least one label
    fp = len(preds.difference(hits))
    # fn are all tags that were never hit
    fn = len(targets.difference(was_hit))
    tn = data_len - tp - fp - fn

    return tp, fp, fn, tn


def _convert_arr_to_windows(array: np.ndarray, window: int, learn_period: int, on_change=True) -> Tuple[
    np.ndarray, Set[Slice]]:
    """ Convert an array of (predictions/labels) to an array of 1s, merging with given window size
        In case of non-spike input, use `on_change=True`

        returns both the new array, and the windows represented as a set of slices """

    if on_change:
        new_array = np.zeros_like(array)
        new_array[1:] = np.abs(array[1:] - array[:-1])
    else:
        new_array = array.copy()

    # disregard learning period
    new_array[:learn_period] = 0
    # close gaps by correlating with box filter and taking the ceil
    new_array = (np.correlate(new_array, np.ones(window * 2 + 1), mode='same') > 0).astype(np.int32)
    # turn into windows as slices
    windows = make_windows(new_array)
    windows = set(map(lambda t: Slice(start=t[0], end=t[1]), windows))

    return new_array, windows


def _calc_pr_re_fb(tp: int, fp: int, fn: int, beta: float = 1.) -> Tuple[float, float, float]:
    """ Calculates and returns precision, recall and Fbeta score from input confusion mat. values """
    b2 = beta ** 2
    pr = tp / (tp + fp + EPS)
    re = tp / (tp + fn + EPS)
    fb = ((1 + b2) * tp) / ((1 + b2) * tp + b2 * fn + fp + EPS)
    return pr, re, fb


def find_best_fb(preds: np.ndarray,
                 target: np.ndarray,
                 thresholds: np.ndarray,
                 label_window_sizes: np.ndarray,
                 learn_period: int,
                 pred_window_size: int = 5,
                 beta: float = 1.) -> Tuple[int, Tuple[float, int]]:
    """
    Find highest Fbeta score, performing a grid search on the different thresholds and label window sizes
    """

    best = 0
    best_params = None
    data_len = len(preds)
    for w_size in label_window_sizes:
        _, target_windows = _convert_arr_to_windows(target, w_size, learn_period=learn_period)
        for thresh in thresholds:
            pred_spikes = (preds > thresh).astype(np.int)
            _, pred_windows = _convert_arr_to_windows(pred_spikes, pred_window_size, on_change=False,
                                                      learn_period=learn_period)
            tp, fp, fn, tn = calc_conf_mat(pred_windows, target_windows, data_len)
            pr, re, fb = _calc_pr_re_fb(tp, fp, fn, beta)
            if fb > best:
                best = fb
                best_params = thresh, w_size
                print(f"Found new best F-score ({beta=:.1f}): {fb:.4f} ({pr=:.2f}, {re=:.2f}) with {best_params}")

    return best, best_params


def merge_targets_and_predictions(predictions: np.ndarray, targets: np.ndarray, thresh: float, window_size: int,
                                  learn_period: int) -> Tuple[np.ndarray, np.ndarray]:

    targets, _ = _convert_arr_to_windows(targets, window_size, learn_period=learn_period)
    pred_spikes = (predictions > thresh).astype(np.int)
    predictions, _ = _convert_arr_to_windows(pred_spikes, 5, on_change=False, learn_period=learn_period)
    return predictions, targets


def plot_windows(pred_merged: np.ndarray, target_merged: np.ndarray, pred_orig: np.ndarray, ground_truth: np.ndarray,
                 thresh: float, learn_period: int):

    learn_period = np.ones(learn_period) * 0.5
    pred_raw = np.zeros_like(pred_orig)
    pred_raw[pred_orig > thresh] = pred_orig[pred_orig > thresh]

    plt.figure(figsize=(15, 5))
    plt.plot(target_merged, label="Anomaly Windows")
    plt.plot(ground_truth, label="Ground Truth")
    plt.plot(learn_period, color='r', label='Learning Period')
    plt.title("Windows around GT")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.plot(target_merged, label="Anomaly Windows")
    plt.plot(pred_merged, label="Anomaly Predictions")
    plt.plot(learn_period, color='r', label='Learning Period')
    plt.title("Final Predictions")
    plt.legend()
    plt.show()


# def _generate_ts(length: int, _start: int, step_seconds: float) -> pd.Series:
#     end = _start + length * step_seconds
#     times = np.arange(_start, end, step_seconds)
#     date_times = map(lambda x: dt.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f'), times)
#     return pd.Series(date_times)
#
#
# def convert_to_NAB(data_df: pd.DataFrame,
#                    al_preds: np.ndarray,
#                    labels: np.ndarray,
#                    ts_col: pd.Series = None,
#                    time_start: int = 1_694_200_000,
#                    step_seconds: float = 1.0) -> pd.DataFrame:
#     if len({len(data_df), len(al_preds), len(labels)}) != 1:
#         raise RuntimeError("all inputs must have the same length")
#
#     if ts_col is None:
#         ts_col = _generate_ts(length=len(data_df), _start=time_start, step_seconds=step_seconds)
#
#     labels, _ = _convert_arr_to_windows(labels, window=0, on_change=True, learn_period=0)
#
#     data_df['value'] = data_df.apply(np.array, 1)
#     data_df['anomaly_score'] = al_preds
#     data_df['timestamp'] = ts_col
#     data_df['label'] = labels
#
#     return data_df[['value', 'anomaly_score', 'timestamp', 'label']].copy()
