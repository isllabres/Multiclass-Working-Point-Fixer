import numpy as np


def compute_proba_f1(y_true: np.ndarray,
                     pred_proba: np.ndarray,
                     working_point: float):
    epsilon = 1e-7

    y_true_reshape = y_true.values.copy().astype(float).ravel()
    y_true_multiclass = np.zeros((y_true_reshape.shape[0], 2))
    y_true_multiclass[:, 1] = y_true_reshape
    y_true_multiclass[:, 0] = 1 - y_true_reshape

    working_point_p = (working_point * 2) - 1
    proportion = (1 + working_point_p) / (1 - working_point_p)
    vector_w = [1., 1.]
    vector_w[1] = vector_w[1] * (1 / (proportion + epsilon))

    y_proba_multiclass = np.ones((pred_proba.shape[0], 2))
    y_proba_multiclass[:, 1] = pred_proba
    y_proba_multiclass[:, 0] = 1 - pred_proba
    y_proba_multiclass_w = y_proba_multiclass.copy() * vector_w
    y_proba_multiclass_w = y_proba_multiclass_w / y_proba_multiclass_w.sum(axis=1).reshape(len(y_proba_multiclass_w), 1)

    p_tp = (y_true_multiclass * y_proba_multiclass_w).sum(axis=0).astype(float)
    p_tn = ((1.0 - y_true_multiclass) * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)
    p_fp = ((1.0 - y_true_multiclass) * y_proba_multiclass_w).sum(axis=0).astype(float)
    p_fn = (y_true_multiclass * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)

    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)

    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)
    p_f1 = p_f1.mean()

    return p_f1