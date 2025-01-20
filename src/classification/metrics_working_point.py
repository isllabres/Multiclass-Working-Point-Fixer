import matplotlib.pyplot as plt
import typing
import pandas
import numpy
from sklearn import metrics

# TODO: Revisar todo este script para mejorarlo y optimizarlo


def predict_with_working_point(
    pred_probability: numpy.ndarray, working_point: float = 0.5
) -> numpy.ndarray:
    """Predict the class for each sample using the indicated probability working point

    Transforms the probabilities into predictions using the working point. Classifying as 'class 1' those probabilities
    upside or equal to the working point value, and as 'class 0' those downside the working point.

    Args:
        pred_probability (numpy.ndarray): the probability of belonging to the minority class for each sample
        working_point (float): The probability threshold

    Returns:
        y_pred (numpy.ndarray): The probability-based predicted class for each of the samples
    """
    if isinstance(working_point, (int, numpy.float64, numpy.float32)):
        working_point = float(working_point)

    y_pred = numpy.zeros((len(pred_probability), 1), dtype=int)
    positions_pred_1 = pred_probability >= working_point
    y_pred[positions_pred_1] = 1

    return y_pred


def compute_metrics(
    y_true: typing.Union[numpy.ndarray, pandas.Series],
    y_pred: typing.Union[numpy.ndarray, pandas.Series],
    verbose: bool = False,
) -> typing.Dict[str, object]:
    """Compute some metrics out of the comparison between y_true and y_pred

    The computed metrics are the following:
        - Confusion matrix values: tn, fp, fn, tp
        - F1-Score: f1
        - Accuracy: acc
        - Average Precision for class 0: ap0
        - Average Precision for class 1: ap1

    Args:
        y_true (typing.Union[numpy.ndarray, pandas.Series]): The ground truth labels
        y_pred (typing.Union[numpy.ndarray, pandas.Series]): The predicted labels
        verbose (bool): If True, print the metrics

    Returns:
        typing.Dict[str, object]: A dictionary containing the computed metrics
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    f1 = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    ap0 = metrics.average_precision_score(y_true, y_pred, pos_label=0)
    ap1 = metrics.average_precision_score(y_true, y_pred, pos_label=1)

    if verbose:
        print(f"Confusion Matrix: tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        print(f"F1-Score: {f1}")
        print(f"Accuracy: {acc}")
        print(f"Average Precision for class 0: {ap0}")
        print(f"Average Precision for class 1: {ap1}")

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1": f1,
        "acc": acc,
        "ap0": ap0,
        "ap1": ap1,
    }


def plot_roc_curve(
    y_true: typing.Union[numpy.ndarray, pandas.Series],
    y_score: typing.Union[numpy.ndarray, pandas.Series],
    title: str = "ROC Curve",
    save_path: str = None,
) -> None:
    """Plot the ROC curve for the given true labels and scores

    Args:
        y_true (typing.Union[numpy.ndarray, pandas.Series]): The ground truth labels
        y_score (typing.Union[numpy.ndarray, pandas.Series]): The predicted scores
        title (str): The title of the plot
        save_path (str): The path to save the plot. If None, the plot is shown but not saved

    Returns:
        None
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true: typing.Union[numpy.ndarray, pandas.Series],
    y_score: typing.Union[numpy.ndarray, pandas.Series],
    title: str = "Precision-Recall Curve",
    save_path: str = None,
) -> None:
    """Plot the Precision-Recall curve for the given true labels and scores

    Args:
        y_true (typing.Union[numpy.ndarray, pandas.Series]): The ground truth labels
        y_score (typing.Union[numpy.ndarray, pandas.Series]): The predicted scores
        title (str): The title of the plot
        save_path (str): The path to save the plot. If None, the plot is shown but not saved

    Returns:
        None
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    average_precision = metrics.average_precision_score(y_true, y_score)

    plt.figure()
    plt.step(recall, precision, where="post", color="b", alpha=0.2, lw=2)
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"{title}: AP={average_precision:0.2f}")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
