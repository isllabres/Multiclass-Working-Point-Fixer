import matplotlib.pyplot as plt
import typing
import pandas
import numpy
import plotly.graph_objects as go
import plotly.io as pio
from sklearn import metrics


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
        y_true (numpy.ndarray): A vector Nx1 containing the true target for N samples
        y_pred (numpy.ndarray): A vector Nx1 containing the predicted target for N samples
        verbose (bool): If True, print the value of the metrics for the optimal and default working points

    Returns:
        tn (int): The number of majority (class 0) samples correctly detected
        fp (int): The number of minority (class 1) samples incorrectly detected
        fn (int): The number of majority (class 0) samples incorrectly detected
        tp (int): The number of majority (class 1)samples correctly detected
        f1 (float): The F1-score value
        acc (float): The Accuracy value
        ap1 (float): The average precision for class 1
        ap0 (float): The average precision for class 0
    """

    label1 = y_true.max().item()
    label0 = y_true.min().item()

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    f1 = metrics.f1_score(
        y_true, y_pred, average="binary", pos_label=label1, zero_division=1
    )
    acc = metrics.accuracy_score(y_true, y_pred)
    ap1 = metrics.average_precision_score(y_true, y_pred, pos_label=label1)
    ap0 = metrics.average_precision_score(y_true, y_pred, pos_label=label0)
    precission = metrics.precision_score(
        y_true, y_pred, average="binary", pos_label=label1, zero_division=1
    )
    recall = metrics.recall_score(
        y_true, y_pred, average="binary", pos_label=label1, zero_division=1
    )

    if verbose:
        print(f"Acc ones:      {numpy.round(tp / (tp + fn), 4)} || tp: {tp} of {tp + fn} ones")
        print(f"Acc zeros:     {numpy.round(tn / (tn + fp), 4)} || tn: {tn} of {tn + fp} ones")
        print(f"Acc:           {numpy.round(acc, 4)}")
        print(f"ap1:           {numpy.round(ap1, 4)}")
        print(f"ap0:           {numpy.round(ap0, 4)}")
        print(f"F1:            {numpy.round(f1, 4)}")
        print(f"Precission:    {numpy.round(precission, 4)}")
        print(f"Recall:        {numpy.round(recall, 4)}")

    dict_metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1": f1,
        "precission": precission,
        "recall": recall,
        "acc": acc,
        "ap1": ap1,
        "ap0": ap0,
    }

    return dict_metrics


def metrics_scanning_working_points_plotly(
    pred_proba: typing.Union[numpy.ndarray, pandas.Series],
    y_true: typing.Union[numpy.ndarray, pandas.Series],
    first_working_point: float,
    last_working_point: float,
    n_working_points: int,
    plot: bool = False,
    figsize: typing.Tuple[int] = (900, 500),
    verbose: bool = False,
) -> typing.Union[float, numpy.ndarray, typing.Dict[str, numpy.ndarray]]:
    """A function to compute a list of metrics for a list of working points

    Iterates the function 'compute_metrics' along the operational range of working points to characterize the behaviour
    of the model in all the working point ranges. Also, if the 'vs_probabilistic_benchmark' is set to True, compute the
    performance of the probabilistic model, and then find a comparable confusion matrix of the model. Comparing both
    confusion matrices is possible to explain the quality of the model decision to the business.

    Args:
        pred_proba (numpy.ndarray): The probability of belonging to the minority class for all the samples
        y_true (numpy.ndarray): The true label for all the samples
        first_working_point (float): The initial value to start the scanning of the working point
        last_working_point (float): The last value to scan the working point
        n_working_points (int): The number of points to scan the working point
        plot (bool): If True, plot the behaviour of the metrics along the scanned working points
        figsize (tuple(int)): The size of the generated figures
        verbose (bool): If True, print the value of the metrics for the optimal and default working points

    Returns:
        working_point_best_f1 (float): The probability working point that maximizes the F1 score
        axis_working_point (numpy.ndarray): The axis of scanned working points
        tn_curve (numpy.ndarray): The number of True Negatives for each scanned working point
        fp_curve (numpy.ndarray): The number of False Positives for each scanned working point
        tp_curve (numpy.ndarray): The number of True Positives for each scanned working point
        f1_curve (numpy.ndarray): The F1-score for each scanned working point
        acc_curve (numpy.ndarray): The Accuracy for each scanned working point
        ap1_curve (numpy.ndarray): The Average Precision for class 1 for each scanned working point
        ap0_curve (numpy.ndarray): The Average Precision for class 0 for each scanned working point
    """
    if isinstance(first_working_point, (int, numpy.float64, numpy.float32)):
        first_working_point = float(first_working_point)
    if isinstance(last_working_point, (int, numpy.float64, numpy.float32)):
        last_working_point = float(last_working_point)

    axis_working_points = numpy.linspace(
        first_working_point, last_working_point, n_working_points
    )

    tn_curve = numpy.zeros(n_working_points)
    fp_curve = numpy.zeros(n_working_points)
    fn_curve = numpy.zeros(n_working_points)
    tp_curve = numpy.zeros(n_working_points)
    f1_curve = numpy.zeros(n_working_points)
    precission_curve = numpy.zeros(n_working_points)
    recall_curve = numpy.zeros(n_working_points)
    acc_curve = numpy.zeros(n_working_points)
    ap1_curve = numpy.zeros(n_working_points)
    ap0_curve = numpy.zeros(n_working_points)
    mean_probability = 0.5

    for index_working_point in range(n_working_points):

        working_point = float(axis_working_points[index_working_point])
        y_pred = predict_with_working_point(
            pred_probability=pred_proba, working_point=working_point
        )
        dict_metrics = compute_metrics(y_true=y_true, y_pred=y_pred, verbose=False)

        tn_curve[index_working_point] = dict_metrics["tn"]
        fp_curve[index_working_point] = dict_metrics["fp"]
        fn_curve[index_working_point] = dict_metrics["fn"]
        tp_curve[index_working_point] = dict_metrics["tp"]
        f1_curve[index_working_point] = dict_metrics["f1"]
        precission_curve[index_working_point] = dict_metrics["precission"]
        recall_curve[index_working_point] = dict_metrics["recall"]
        acc_curve[index_working_point] = dict_metrics["acc"]
        ap1_curve[index_working_point] = dict_metrics["ap1"]
        ap0_curve[index_working_point] = dict_metrics["ap0"]

    pos_best_f1 = numpy.where(f1_curve == f1_curve.max())[0][0]
    working_point_best_f1 = axis_working_points[pos_best_f1]

    def abs_difference_mean(list_value):
        return abs(list_value - mean_probability)
    pos_default_working_point = numpy.where(
        axis_working_points == min(axis_working_points, key=abs_difference_mean)
    )[0][0]

    if verbose:
        print(f"Metrics for 'default' working point = {mean_probability}")
        print(
            f"\t tn   {int(tn_curve[pos_default_working_point])} \t//fp {int(fp_curve[pos_default_working_point])}"
            f"\t//fn {int(fn_curve[pos_default_working_point])} \t//tp {int(tp_curve[pos_default_working_point])}"
        )
        print(
            f"\t Ap0: {numpy.round(ap0_curve[pos_default_working_point],4)} "
            f"\t //Ap1:{numpy.round(ap1_curve[pos_default_working_point],4)}"
        )
        print(f"\t f1:         {numpy.round(f1_curve[pos_default_working_point],4)}")
        print(
            f"\t Precission: {numpy.round(precission_curve[pos_default_working_point],4)}"
        )
        print(
            f"\t Recall:     {numpy.round(recall_curve[pos_default_working_point],4)}"
        )
        print(f"\t Acc:        {numpy.round(acc_curve[pos_default_working_point],4)}")

        print(f"Metrics for 'best' working point = {working_point_best_f1}")
        print(
            f"\t tn   {int(tn_curve[pos_best_f1])} \t//fp {int(fp_curve[pos_best_f1])}"
            f"\t//fn {int(fn_curve[pos_best_f1])} \t//tp {int(tp_curve[pos_best_f1])}"
        )
        print(
            f"\t Ap0: {numpy.round(ap0_curve[pos_best_f1],4)} \t //Ap1:{numpy.round(ap1_curve[pos_best_f1],4)}"
        )
        print(f"\t f1:         {numpy.round(f1_curve[pos_best_f1],4)}")
        print(f"\t Precission: {numpy.round(precission_curve[pos_best_f1],4)}")
        print(f"\t Recall:     {numpy.round(recall_curve[pos_best_f1],4)}")
        print(f"\t Acc:        {numpy.round(acc_curve[pos_best_f1],4)}")

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=axis_working_points, y=f1_curve, mode='lines', name='F1 Score'))
        fig.add_trace(go.Scatter(x=axis_working_points, y=precission_curve, mode='lines', name='Precision'))
        fig.add_trace(go.Scatter(x=axis_working_points, y=recall_curve, mode='lines', name='Recall'))

        fig.add_trace(go.Scatter(x=[axis_working_points[pos_default_working_point]],
                                 y=[numpy.round(f1_curve[pos_default_working_point], 4)],
                                 mode='markers', name='F1 for default WP',
                                 marker=dict(color='red', size=7, symbol='x')))
        fig.add_trace(go.Scatter(x=[axis_working_points[pos_default_working_point]],
                                 y=[numpy.round(precission_curve[pos_default_working_point], 4)],
                                 mode='markers', name='Precision for default WP',
                                 marker=dict(color='red', size=7, symbol='x')))
        fig.add_trace(go.Scatter(x=[axis_working_points[pos_default_working_point]],
                                 y=[numpy.round(recall_curve[pos_default_working_point], 4)],
                                 mode='markers', name='Recall for default WP',
                                 marker=dict(color='red', size=7, symbol='x')))
        fig.add_trace(go.Scatter(x=[axis_working_points[pos_best_f1]],
                                 y=[numpy.round(f1_curve[pos_best_f1], 4)],
                                 mode='markers', name='F1 for best WP',
                                 marker=dict(color='blue', size=7)))
        fig.add_trace(go.Scatter(x=[axis_working_points[pos_best_f1]],
                                 y=[numpy.round(precission_curve[pos_best_f1], 4)],
                                 mode='markers', name='Precision for best WP',
                                 marker=dict(color='blue', size=7)))
        fig.add_trace(go.Scatter(x=[axis_working_points[pos_best_f1]],
                                 y=[numpy.round(recall_curve[pos_best_f1], 4)],
                                 mode='markers', name='Recall for best WP',
                                 marker=dict(color='blue', size=7)))
        fig.update_layout(
            title=f'Metrics Scanning Working Points.'\
                  f' Default WP: {numpy.round(axis_working_points[pos_default_working_point], 3)}, '
                  f' F1 {numpy.round(f1_curve[pos_default_working_point], 3)}.'\
                  f' Best WP: {numpy.round(axis_working_points[pos_best_f1], 3)},'\
                  f' F1 {numpy.round(f1_curve[pos_best_f1], 3)}.',
            xaxis_title='Working Point',
            yaxis_title='Metric Value',
            width=figsize[0],
            height=figsize[1]
        )

        pio.renderers.default = 'notebook'
        fig.show()

    dict_metrics_curve = {
        "tn_curve": tn_curve,
        "fp_curve": fp_curve,
        "fn_curve": fn_curve,
        "tp_curve": tp_curve,
        "f1_curve": f1_curve,
        "precission_curve": precission_curve,
        "recall_curve": recall_curve,
        "acc_curve": acc_curve,
        "ap1_curve": ap1_curve,
        "ap0_curve": ap0_curve,
    }

    results = {
        "working_point_best_f1": working_point_best_f1,
        "axis_working_points": axis_working_points,
        "dict_metrics_curve": dict_metrics_curve,
    }

    return results


def metrics_scanning_working_points_matplotlib(
    pred_proba: typing.Union[numpy.ndarray, pandas.Series],
    y_true: typing.Union[numpy.ndarray, pandas.Series],
    first_working_point: float,
    last_working_point: float,
    n_working_points: int,
    plot: bool = False,
    figsize: typing.Tuple[int] = (9, 5),
    verbose: bool = False,
) -> typing.Union[float, numpy.ndarray, typing.Dict[str, numpy.ndarray]]:
    """A function to compute a list of metrics for a list of working points

    Iterates the function 'compute_metrics' along the operational range of working points to characterize the behaviour
    of the model in all the working point ranges. Also, if the 'vs_probabilistic_benchmark' is set to True, compute the
    performance of the probabilistic model, and then find a comparable confusion matrix of the model. Comparing both
    confusion matrices is possible to explain the quality of the model decision to the business.

    Args:
        pred_proba (numpy.ndarray): The probability of belonging to the minority class for all the samples
        y_true (numpy.ndarray): The true label for all the samples
        first_working_point (float): The initial value to start the scanning of the working point
        last_working_point (float): The last value to scan the working point
        n_working_points (int): The number of points to scan the working point
        plot (bool): If True, plot the behaviour of the metrics along the scanned working points
        figsize (tuple(int)): The size of the generated figures
        verbose (bool): If True, print the value of the metrics for the optimal and default working points
        vs_probabilistic_benchmark (bool): If True, compute a comparable confusion matrix for the probabilistic method
                                           and the model

    Returns:
        working_point_best_f1 (float): The probability working point that maximizes the F1 score
        axis_working_point (numpy.ndarray): The axis of scanned working points
        tn_curve (numpy.ndarray): The number of True Negatives for each scanned working point
        fp_curve (numpy.ndarray): The number of False Positives for each scanned working point
        tp_curve (numpy.ndarray): The number of True Positives for each scanned working point
        f1_curve (numpy.ndarray): The F1-score for each scanned working point
        acc_curve (numpy.ndarray): The Accuracy for each scanned working point
        ap1_curve (numpy.ndarray): The Average Precision for class 1 for each scanned working point
        ap0_curve (numpy.ndarray): The Average Precision for class 0 for each scanned working point
    """
    if isinstance(first_working_point, (int, numpy.float64, numpy.float32)):
        first_working_point = float(first_working_point)
    if isinstance(last_working_point, (int, numpy.float64, numpy.float32)):
        last_working_point = float(last_working_point)

    axis_working_points = numpy.linspace(
        first_working_point, last_working_point, n_working_points
    )

    tn_curve = numpy.zeros(n_working_points)
    fp_curve = numpy.zeros(n_working_points)
    fn_curve = numpy.zeros(n_working_points)
    tp_curve = numpy.zeros(n_working_points)
    f1_curve = numpy.zeros(n_working_points)
    precission_curve = numpy.zeros(n_working_points)
    recall_curve = numpy.zeros(n_working_points)
    acc_curve = numpy.zeros(n_working_points)
    ap1_curve = numpy.zeros(n_working_points)
    ap0_curve = numpy.zeros(n_working_points)
    mean_probability = 0.5

    for index_working_point in range(n_working_points):

        working_point = float(axis_working_points[index_working_point])
        y_pred = predict_with_working_point(
            pred_probability=pred_proba, working_point=working_point
        )
        dict_metrics = compute_metrics(y_true=y_true, y_pred=y_pred, verbose=False)

        tn_curve[index_working_point] = dict_metrics["tn"]
        fp_curve[index_working_point] = dict_metrics["fp"]
        fn_curve[index_working_point] = dict_metrics["fn"]
        tp_curve[index_working_point] = dict_metrics["tp"]
        f1_curve[index_working_point] = dict_metrics["f1"]
        precission_curve[index_working_point] = dict_metrics["precission"]
        recall_curve[index_working_point] = dict_metrics["recall"]
        acc_curve[index_working_point] = dict_metrics["acc"]
        ap1_curve[index_working_point] = dict_metrics["ap1"]
        ap0_curve[index_working_point] = dict_metrics["ap0"]

    pos_best_f1 = numpy.where(f1_curve == f1_curve.max())[0][0]
    working_point_best_f1 = axis_working_points[pos_best_f1]

    def abs_difference_mean(list_value):
        return abs(list_value - mean_probability)

    pos_default_working_point = numpy.where(
        axis_working_points == min(axis_working_points, key=abs_difference_mean)
    )[0][0]

    if verbose:

        print(f"Metrics for 'default' working point = {mean_probability}")
        print(
            f"\t tn   {int(tn_curve[pos_default_working_point])} \t//fp {int(fp_curve[pos_default_working_point])}"
            f"\t//fn {int(fn_curve[pos_default_working_point])} \t//tp {int(tp_curve[pos_default_working_point])}"
        )
        print(
            f"\t Ap0: {numpy.round(ap0_curve[pos_default_working_point],4)}"
            + f"\t //Ap1:{numpy.round(ap1_curve[pos_default_working_point],4)}"
        )
        print(f"\t f1:         {numpy.round(f1_curve[pos_default_working_point],4)}")
        print(
            f"\t Precission: {numpy.round(precission_curve[pos_default_working_point],4)}"
        )
        print(
            f"\t Recall:     {numpy.round(recall_curve[pos_default_working_point],4)}"
        )
        print(f"\t Acc:        {numpy.round(acc_curve[pos_default_working_point],4)}")

        print(f"Metrics for max_f1 working point = {working_point_best_f1}")
        print(
            f"\t tn   {int(tn_curve[pos_best_f1])} \t//fp {int(fp_curve[pos_best_f1])}"
            f"\t//fn {int(fn_curve[pos_best_f1])} \t//tp {int(tp_curve[pos_best_f1])}"
        )
        print(
            f"\t Ap0: {numpy.round(ap0_curve[pos_best_f1],4)} \t //Ap1:{numpy.round(ap1_curve[pos_best_f1],4)}"
        )
        print(f"\t f1:         {numpy.round(f1_curve[pos_best_f1],4)}")
        print(f"\t Precission: {numpy.round(precission_curve[pos_best_f1],4)}")
        print(f"\t Recall:     {numpy.round(recall_curve[pos_best_f1],4)}")
        print(f"\t Acc:        {numpy.round(acc_curve[pos_best_f1],4)}")

    if plot:

        plt.figure(figsize=figsize)
        plt.xlabel("working_point")

        plt.plot(
            axis_working_points[pos_default_working_point],
            f1_curve[pos_default_working_point],
            "*r",
            label="default_wp",
        )
        plt.plot(
            axis_working_points[pos_best_f1],
            f1_curve[pos_best_f1],
            "ob",
            label="max_f1_wp",
        )
        plt.plot(axis_working_points, f1_curve, label="F1 Score")

        plt.plot(
            axis_working_points[pos_default_working_point],
            precission_curve[pos_default_working_point],
            "*r",
        )
        plt.plot(axis_working_points[pos_best_f1], precission_curve[pos_best_f1], "ob")
        plt.plot(axis_working_points, precission_curve, label="Precission")

        plt.plot(
            axis_working_points[pos_default_working_point],
            recall_curve[pos_default_working_point],
            "*r",
        )
        plt.plot(axis_working_points[pos_best_f1], recall_curve[pos_best_f1], "ob")
        plt.plot(axis_working_points, recall_curve, label="Recall")

        plt.legend()
        plt.show()

    dict_metrics_curve = {
        "tn_curve": tn_curve,
        "fp_curve": fp_curve,
        "fn_curve": fn_curve,
        "tp_curve": tp_curve,
        "f1_curve": f1_curve,
        "precission_curve": precission_curve,
        "recall_curve": recall_curve,
        "acc_curve": acc_curve,
        "ap1_curve": ap1_curve,
        "ap0_curve": ap0_curve,
    }

    results = {
        "working_point_best_f1": working_point_best_f1,
        "axis_working_points": axis_working_points,
        "dict_metrics_curve": dict_metrics_curve,
    }

    return results
