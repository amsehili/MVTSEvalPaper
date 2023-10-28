from collections import namedtuple
import numpy as np

EventWiseMetrics = namedtuple(
    "EventWiseMetrics", "TP_ew FP_ew FN_ew P_ew R_ew F1_ew F1_c"
)


def make_intervals(y):
    """Find intervals of consecutive 1 in input array.

    Args:
        y (array): input array of labels or predictions. Should
        be an array of bool or 0/1 values.

    Returns:
        list: list of intervals, each as a (onset, offset) tuple.
    """
    y = np.asarray(y).astype(int)
    d = np.diff(y, prepend=0, append=0)
    (onsets,) = np.where(d == 1)
    (offsets,) = np.where(d == -1)
    return list(zip(onsets, offsets))


def compute_event_wise_metrics(y_true, y_pred, gt_intervals=None):
    """Compute event-wise metrics including composite F1 score.


    Args:
        y_true (array): ground truth anomalies.
        y_pred (array): predicted anomalies.
        gt_intervals (list, optional): ground truth intervals of anomalies
            (i.e., anomalous event). Although most of the time  this can be
            automatically computed using `y_true`, in some cases you'd want
            to explicitly pass the events to avoid that two contiguous but
            independent events be considered as one single event.

    Returns:
        namedtuple: an object with the following fields containing event-wise
        metrics: TP_ew, FP_ew, FN_ew, P_ew, R_ew, F1_ew and F1_c.
    """

    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=int)
    intervals = make_intervals(y_true) if gt_intervals is None else gt_intervals
    pred_intervals = make_intervals(y_pred)

    y_pred_proc = y_pred.copy()

    TP = 0
    FP = 0
    FN = 0

    # Count correctly detected events
    for onset, offset in intervals:
        if y_pred[onset:offset].any():
            TP += 1
        else:
            FN += 1

    # Count segments that don't overlap with any ground truth event

    for onset, offset in pred_intervals:
        if (~y_true[onset:offset]).all():
            FP += 1

    # Compute 1 - False Alarm Rate point-wise
    nFPR = 1 - ((~y_true) & (y_pred == 1)).sum() / (~y_true).sum()

    # Compute event-wise Precision, Recall and F1 score
    P_ew = TP / (TP + FP) * nFPR
    R_ew = TP / (TP + FN)
    F1_ew = (2 * P_ew * R_ew) / (P_ew + R_ew)

    # Compute composite F1 score
    P = ((y_true) & (y_pred == 1)).sum() / y_pred.sum()  # point-wise precision
    F1_c = 2 * P * R_ew / (P + R_ew)
    return EventWiseMetrics(TP, FP, FN, P_ew, R_ew, F1_ew, F1_c)


def smooth_scores(scores, history_size=3):
    """Smooth input score values using a simple moving average. Each value
    `s_i` in `scores` will be replaced by `mean(scores[i-history_size:i+1].
    Note that the first `history_size` values of `scores` are returned
    unchanged.

    Args:
        scores (array): array of scores.
        history_size (int, optional): number of history values to consider
            for average value calculation. Defaults to 3.

    Returns:
        array: smoothed scores.
    """
    N = history_size + 1
    smoothed = np.convolve(np.ones(N) / N, scores, mode="valid")
    # The first `history_size` points are not affected
    return np.concatenate([scores[:history_size], smoothed])


def predict_with_PCA(
    pca, X, add_mean=False, score_agg_fn=np.max, smooth_n=None
):
    """Compute anomaly scores through the following steps:
        - Transform X with model
        - Perform an inverse transform to obtain a reconstruction of X
        - Compute the squared difference between X and its reconstruction
        - Take the maximum of the squared difference for each timestamp
        - Optionally smooth scores

    Args:
        pca (Estimator): fitted PCA model.
        X (array): input array of shape (T,N) where T is the number
            of timestamps and N is the number of features.
        add_mean (bool, optional): add data mean after project on eigenvectors'
            matrix (equivalent to pca.inverse_transform). Default=False.
        score_agg_fn (callable, optional): function to aggregate residual
            features from all channels. Default: numpy.max.
        smooth_n (int, optional): history size of score smoothing.
            if not specified (defaults), no smoothing is performed.

    Returns:
        array: array of anomaly scores of shape (T,).
    """
    X_tran = pca.transform(X)
    X_pred = np.dot(X_tran, pca.components_)
    if add_mean:
        # becomes equivalent to pca.inverse_transform(X_tran)
        X_pred += pca.mean_

    X_err = (X_pred - X) ** 2
    scores = score_agg_fn(X_err, axis=1)
    if smooth_n:
        return smooth_scores(scores, smooth_n)
    else:
        return scores
