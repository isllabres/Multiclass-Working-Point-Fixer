import numpy
import torch
import pandas


def soft_p_f1_loss(target: torch.Tensor,
                   model_output: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-7

    p_tp = (target * model_output).sum(axis=0).to(torch.float32)
    p_fp = ((1.0 - target) * model_output).sum(axis=0).to(torch.float32)
    p_fn = (target * (1.0 - model_output)).sum(axis=0).to(torch.float32)

    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)

    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    p_f1 = torch.mean(p_f1)  # Mean to compute the F1 score "average='macro'"

    return 1 - p_f1


def soft_p_f1_loss_wp(
    model_output: torch.Tensor, target: torch.Tensor, working_point: float
) -> torch.Tensor:

    epsilon = 1e-7

    target_multiclass = torch.zeros((target.shape[0], 2))
    target_multiclass[:, 1] = target
    target_multiclass[:, 0] = torch.sub(1, target)

    working_point_p = torch.sub(torch.mul(working_point, 2), 1)
    proportion = torch.div(torch.add(1, working_point_p), torch.sub(1, working_point_p))

    vector_w = torch.ones(1, 2)
    vector_w[:, 1] = torch.mul(
        vector_w[:, 1], torch.div(1, torch.add(proportion, epsilon))
    )

    model_output_multiclass = torch.ones((model_output.shape[0], 2))
    model_output_multiclass[:, 1] = model_output
    model_output_multiclass[:, 0] = torch.sub(1, model_output)

    model_output_multiclass_w = torch.mul(model_output_multiclass, vector_w)
    model_output_multiclass_w = torch.div(
        model_output_multiclass_w,
        model_output_multiclass_w.sum(axis=1).reshape(
            len(model_output_multiclass_w), 1
        ),
    )

    p_tp = (target_multiclass * model_output_multiclass_w).sum(axis=0).to(torch.float32)
    p_fp = (
        ((1.0 - target_multiclass) * model_output_multiclass_w)
        .sum(axis=0)
        .to(torch.float32)
    )
    p_fn = (
        (target_multiclass * (1.0 - model_output_multiclass_w))
        .sum(axis=0)
        .to(torch.float32)
    )

    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)

    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    p_f1 = p_f1.mean()  # Mean to compute the F1 score "average='macro'"

    return 1 - p_f1


def compute_proba_f1(y_true: numpy.ndarray, pred_proba: numpy.ndarray, working_point: float):
    epsilon = 1e-7

    y_true_reshape = y_true.values.copy().astype(float).ravel()
    y_true_multiclass = numpy.zeros((y_true_reshape.shape[0], 2))
    y_true_multiclass[:, 1] = y_true_reshape
    y_true_multiclass[:, 0] = 1 - y_true_reshape

    working_point_p = (working_point * 2) - 1
    proportion = (1 + working_point_p) / (1 - working_point_p)
    vector_w = [1.0, 1.0]
    vector_w[1] = vector_w[1] * (1 / (proportion + epsilon))

    y_proba_multiclass = numpy.ones((pred_proba.shape[0], 2))
    y_proba_multiclass[:, 1] = pred_proba
    y_proba_multiclass[:, 0] = 1 - pred_proba
    y_proba_multiclass_w = y_proba_multiclass.copy() * vector_w
    y_proba_multiclass_w = y_proba_multiclass_w / y_proba_multiclass_w.sum(
        axis=1
    ).reshape(len(y_proba_multiclass_w), 1)

    p_tp = (y_true_multiclass * y_proba_multiclass_w).sum(axis=0).astype(float)
    p_fp = ((1.0 - y_true_multiclass) * y_proba_multiclass_w).sum(axis=0).astype(float)
    p_fn = (y_true_multiclass * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)

    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)

    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)
    p_f1 = p_f1.mean()

    return p_f1


def compute_pf1_scores(y_true, y_pred, th):
    """
    This function calculates two versions of the probabilistic F1 score:
    1. Using the torch library to compute the score.
    2. Using a custom implementation to compute the score.
    Args:
        y_true (numpy.ndarray): The ground truth labels.
        y_pred (numpy.ndarray): The predicted probabilities.
        th (float): The working point threshold.
    Returns:
        tuple: A tuple containing:
            - p_f1_t (float): The probabilistic F1 score computed using the torch library.
            - p_f1 (float): The probabilistic F1 score computed using a custom implementation.
    """

    p_f1_t = -(
        soft_p_f1_loss_wp(
            model_output=torch.from_numpy(y_pred),
            target=torch.from_numpy(y_true),
            working_point=th,
        )
        - 1
    )  # undo loss calculation 1-p_f1
    p_f1 = compute_proba_f1(pandas.DataFrame(y_true), y_pred, working_point=th)

    return p_f1_t, p_f1
