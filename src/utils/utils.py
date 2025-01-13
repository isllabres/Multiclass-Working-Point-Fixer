import typing
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

import matplotlib.pyplot
import numpy

from REPCLASS.src.repclass.pricing.hard_optimizers.probability_models.probability_metrics import predict_with_working_point
from REPCLASS.src.repclass.pricing.hard_optimizers.probability_models.monotonic_LGBM import monotonic_LGBM
from REPCLASS.src.repclass.utils.auxiliar_functions import Columns

def predict_with_working_point(pred_probability: numpy.ndarray, working_point: float = 0.0) -> numpy.ndarray:
    """Predict the class for each sample using the indicated probability working point

    Transforms the probabilities into predictions using the working point. Classifying as 'class 1' those probabilities
    upside or equal to the working point value, and as 'class 0' those downside the working point.

    Args:
        pred_probability (numpy.ndarray): the probability of belonging to the minority class for each sample
        working_point (float): The probability threshold

    Returns:
        y_pred (numpy.ndarray): The probability-based predicted class for each of the samples
    """
    y_pred = numpy.zeros((len(pred_probability), 1), dtype=int)
    positions_pred_1 = pred_probability >= working_point
    y_pred[positions_pred_1] = 1
    return y_pred


def compute_proba_f1_multiclass(y_true: numpy.ndarray, pred_proba: numpy.ndarray, working_point: float):
    epsilon = 1e-7
    y_true_reshape = y_true.values.copy().astype(float).ravel()
    y_true_multiclass = numpy.zeros((y_true_reshape.shape[0], 2))
    y_true_multiclass[:, 1] = y_true_reshape
    y_true_multiclass[:, 0] = 1 - y_true_reshape
    working_point_p = (working_point * 2) - 1
    proportion = (1 + working_point_p) / (1 - working_point_p)
    vector_w = [1., 1.]
    vector_w[1] = vector_w[1] * (1 / (proportion + epsilon))
    y_proba_multiclass = numpy.ones((pred_proba.shape[0], 2))
    y_proba_multiclass[:, 1] = pred_proba
    y_proba_multiclass[:, 0] = 1 - pred_proba
    y_proba_multiclass_w = y_proba_multiclass.copy() * vector_w
    y_proba_multiclass_w = y_proba_multiclass_w / y_proba_multiclass_w.sum(axis=1).reshape(len(y_proba_multiclass_w), 1)
    """p_tp = (y_true_multiclass * y_pred).sum(axis=0).astype(float)#.to(torch.float32)
    p_tn = ((1.0 - y_true_multiclass) * (1.0 - y_pred)).sum(axis=0).astype(float)#.to(torch.float32)
    p_fp = ((1.0 - y_true_multiclass) * y_pred).sum(axis=0).astype(float)#.to(torch.float32)
    p_fn = (y_true_multiclass * (1.0 - y_pred)).sum(axis=0).astype(float)#.to(torch.float32)"""
    p_tp = (y_true_multiclass * y_proba_multiclass_w).sum(axis=0).astype(float)  # .to(torch.float32)
    p_tn = ((1.0 - y_true_multiclass) * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)  # .to(torch.float32)
    p_fp = ((1.0 - y_true_multiclass) * y_proba_multiclass_w).sum(axis=0).astype(float)  # .to(torch.float32)
    p_fn = (y_true_multiclass * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)  # .to(torch.float32)
    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)
    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)
    print()
    p_f1 = p_f1[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_precision = p_precision[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_recall = p_recall.mean()  # [1]#.mean()  # Mean to compute the F1 score "average='macro'"
    p_tp = p_tp[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_tn = p_tn[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_fp = p_fp[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_fn = p_fn[1]  # .mean()  # Mean to compute the F1 score "average='macro'"

    return p_f1, p_precision, p_recall, p_tp, p_tn, p_fp, p_fn

def compute_proba_f1(y_true: numpy.ndarray, pred_proba: numpy.ndarray, working_point: float):
    epsilon = 1e-7
    y_true_reshape = y_true.values.copy().astype(float).ravel()
    y_true_multiclass = numpy.zeros((y_true_reshape.shape[0], 2))
    y_true_multiclass[:, 1] = y_true_reshape
    y_true_multiclass[:, 0] = 1 - y_true_reshape
    working_point_p = (working_point * 2) - 1
    proportion = (1 + working_point_p) / (1 - working_point_p)
    vector_w = [1., 1.]
    vector_w[1] = vector_w[1] * (1 / (proportion + epsilon))
    y_proba_multiclass = numpy.ones((pred_proba.shape[0], 2))
    y_proba_multiclass[:, 1] = pred_proba
    y_proba_multiclass[:, 0] = 1 - pred_proba
    y_proba_multiclass_w = y_proba_multiclass.copy() * vector_w
    y_proba_multiclass_w = y_proba_multiclass_w / y_proba_multiclass_w.sum(axis=1).reshape(len(y_proba_multiclass_w), 1)
    """p_tp = (y_true_multiclass * y_pred).sum(axis=0).astype(float)#.to(torch.float32)
    p_tn = ((1.0 - y_true_multiclass) * (1.0 - y_pred)).sum(axis=0).astype(float)#.to(torch.float32)
    p_fp = ((1.0 - y_true_multiclass) * y_pred).sum(axis=0).astype(float)#.to(torch.float32)
    p_fn = (y_true_multiclass * (1.0 - y_pred)).sum(axis=0).astype(float)#.to(torch.float32)"""
    p_tp = (y_true_multiclass * y_proba_multiclass_w).sum(axis=0).astype(float)  # .to(torch.float32)
    p_tn = ((1.0 - y_true_multiclass) * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)  # .to(torch.float32)
    p_fp = ((1.0 - y_true_multiclass) * y_proba_multiclass_w).sum(axis=0).astype(float)  # .to(torch.float32)
    p_fn = (y_true_multiclass * (1.0 - y_proba_multiclass_w)).sum(axis=0).astype(float)  # .to(torch.float32)

    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)
    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)
    print()
    p_f1 = p_f1[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_precision = p_precision[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_recall = p_recall.mean()  # [1]#.mean()  # Mean to compute the F1 score "average='macro'"
    p_tp = p_tp[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_tn = p_tn[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_fp = p_fp[1]  # .mean()  # Mean to compute the F1 score "average='macro'"
    p_fn = p_fn[1]  # .mean()  # Mean to compute the F1 score "average='macro'"

    if numpy.isnan(p_f1):
        print("ERROR")
    return p_f1, p_precision, p_recall, p_tp, p_tn, p_fp, p_fn

def compute_metrics_new(y_true: numpy.ndarray, y_pred: numpy.ndarray, pred_proba: numpy.ndarray, working_point: float,
                        verbose=False) -> typing.Dict[str, object]:
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
    p_f1, p_precision, p_recall, p_tp, p_tn, p_fp, p_fn = compute_proba_f1(y_true=y_true, pred_proba=pred_proba,
                                                                           working_point=working_point)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='binary', pos_label=label1, zero_division=1)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    ap1 = sklearn.metrics.average_precision_score(y_true, y_pred, pos_label=label1)
    ap0 = sklearn.metrics.average_precision_score(y_true, y_pred, pos_label=label0)
    precission = sklearn.metrics.precision_score(y_true, y_pred, average='binary', pos_label=label1, zero_division=1)
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='binary', pos_label=label1, zero_division=1)
    if verbose:
        print(f"Acc ones:      {numpy.round(tp / (tp + fn), 4)} || tp: {tp} of {tp + fn} ones")
        print(f"Acc zeros:     {numpy.round(tn / (tn + fp), 4)} || tn: {tn} of {tn + fp} ones")
        print(f"Acc:           {numpy.round(acc, 4)}")
        print(f"ap1:           {numpy.round(ap1, 4)}")
        print(f"ap0:           {numpy.round(ap0, 4)}")
        print(f"F1:            {numpy.round(f1, 4)}")
        print(f"Precission:    {numpy.round(precission, 4)}")
        print(f"Recall:        {numpy.round(recall, 4)}")
    dict_metrics = {"tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "f1": f1,
                    "p_f1": p_f1,
                    "p_precision": p_precision,
                    "p_recall": p_recall,
                    "p_tn": p_tn,
                    "p_fp": p_fp,
                    "p_fn": p_fn,
                    "p_tp": p_tp,
                    "precission": precission,
                    "recall": recall,
                    "acc": acc,
                    "ap1": ap1,
                    "ap0": ap0}

    return dict_metrics


def metrics_scanning_working_points_new(pred_proba: numpy.ndarray,
                                        y_true: numpy.ndarray,
                                        imbalance_ratio: float,
                                        first_working_point: float,
                                        last_working_point: float,
                                        n_working_points: int,
                                        fixed_best_working_point: float = None,
                                        fixed_best_working_point_p_f1: float = None,
                                        plot: bool = False,
                                        verbose: bool = False,
                                        vs_probabilistic_benchmark: bool = False
                                        ) -> typing.Union[float, numpy.ndarray, typing.Dict[str, numpy.ndarray]]:
    """A function to compute a list of metrics for a list of working points

    Iterates the function 'compute_metrics' along the operational range of working points to characterize the behaviour
    of the model in all the working point ranges. Also, if the 'vs_probabilistic_benchmark' is set to True, compute the
    performance of the probabilistic model, and then find a comparable confusion matrix of the model. Comparing both
    confusion matrices is possible to explain the quality of the model decision to the business.

    Args:
        pred_proba (numpy.ndarray): The probability of belonging to the minority class for all the samples
        y_true (numpy.ndarray): The true label for all the samples
        imbalance_ratio (float): The imbalance ratio of the training set
        ini_working_point (float): The initial value to start the scanning of the working point
        step_working_point (float): The step size value to scan the working point
        n_working_points (int): The number of points to scan the working point
        plot (bool): If True, plot the behaviour of the metrics along the scanned working points
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
    axis_working_points = numpy.linspace(first_working_point, last_working_point, n_working_points)
    tn_curve = numpy.zeros(n_working_points)
    fp_curve = numpy.zeros(n_working_points)
    fn_curve = numpy.zeros(n_working_points)
    tp_curve = numpy.zeros(n_working_points)
    f1_curve = numpy.zeros(n_working_points)
    p_f1_curve = numpy.zeros(n_working_points)
    p_precision_curve = numpy.zeros(n_working_points)
    p_recall_curve = numpy.zeros(n_working_points)
    p_tn_curve = numpy.zeros(n_working_points)
    p_fp_curve = numpy.zeros(n_working_points)
    p_fn_curve = numpy.zeros(n_working_points)
    p_tp_curve = numpy.zeros(n_working_points)
    precission_curve = numpy.zeros(n_working_points)
    recall_curve = numpy.zeros(n_working_points)
    acc_curve = numpy.zeros(n_working_points)
    ap1_curve = numpy.zeros(n_working_points)
    ap0_curve = numpy.zeros(n_working_points)
    mean_probability = 0.5
    for index_working_point in range(n_working_points):
        working_point = axis_working_points[index_working_point]
        y_pred = predict_with_working_point(pred_probability=pred_proba, working_point=working_point)
        dict_metrics = compute_metrics_new(y_true=y_true, y_pred=y_pred, pred_proba=pred_proba,
                                           working_point=working_point, verbose=False)
        tn_curve[index_working_point] = dict_metrics["tn"]
        fp_curve[index_working_point] = dict_metrics["fp"]
        fn_curve[index_working_point] = dict_metrics["fn"]
        tp_curve[index_working_point] = dict_metrics["tp"]
        f1_curve[index_working_point] = dict_metrics["f1"]
        p_f1_curve[index_working_point] = dict_metrics["p_f1"]
        p_precision_curve[index_working_point] = dict_metrics["p_precision"]
        p_recall_curve[index_working_point] = dict_metrics["p_recall"]
        p_tn_curve[index_working_point] = dict_metrics["p_tn"]
        p_fp_curve[index_working_point] = dict_metrics["p_fp"]
        p_fn_curve[index_working_point] = dict_metrics["p_fn"]
        p_tp_curve[index_working_point] = dict_metrics["p_tp"]
        precission_curve[index_working_point] = dict_metrics["precission"]
        recall_curve[index_working_point] = dict_metrics["recall"]
        acc_curve[index_working_point] = dict_metrics["acc"]
        ap1_curve[index_working_point] = dict_metrics["ap1"]
        ap0_curve[index_working_point] = dict_metrics["ap0"]
    if fixed_best_working_point is None and fixed_best_working_point_p_f1 is None:
        # pos_best_f1 = numpy.where(f1_curve == f1_curve.max())[0][0]
        pos_best_f1 = numpy.where(f1_curve == f1_curve.max())[0][0]
        pos_best_p_f1 = numpy.where(p_f1_curve == p_f1_curve.max())[0][0]
        working_point_best_f1 = axis_working_points[pos_best_f1]
        working_point_best_p_f1 = axis_working_points[pos_best_p_f1]
    elif type(fixed_best_working_point) != type(fixed_best_working_point_p_f1):
        ValueError("fixed_best_working_point and fixed_best_working_point_p_f1 should be both None or float")
    else:
        def abs_difference_best(list_value):
            return abs(list_value - fixed_best_working_point)
        def abs_difference_best_p_f1(list_value):
            return abs(list_value - fixed_best_working_point_p_f1)
        # abs_difference = lambda list_value: abs(list_value - fixed_best_working_point)
        pos_best_f1 = numpy.where(axis_working_points == min(axis_working_points, key=abs_difference_best))[0][0]
        pos_best_p_f1 = numpy.where(axis_working_points == min(axis_working_points, key=abs_difference_best_p_f1))[0][0]
        working_point_best_f1 = axis_working_points[pos_best_f1]
        working_point_best_p_f1 = axis_working_points[pos_best_p_f1]

    def abs_difference_mean(list_value):
        return abs(list_value - mean_probability)
    # abs_difference_mean = lambda list_value: abs(list_value - mean_probability)
    pos_default_working_point = numpy.where(axis_working_points == min(axis_working_points,
                                                                       key=abs_difference_mean))[0][0]
    if vs_probabilistic_benchmark:
       # ########## Metrics of the probabilistic ########## #
        # Add 1 to imbalance_ratio, because the IB = nºminor/nºmayor, and probability os class 0 is
        # equal to (nºminor/nºtot) = (nºminor/(nºmayor+nºminor))
        prob_1 = 1 / (imbalance_ratio + 1)
        prob_0 = 1 - prob_1
        n_samples = len(y_true)
        tn_probabilistic = int(prob_0 * prob_0 * n_samples)
        fp_probabilistic = int(prob_1 * prob_0 * n_samples)
        fn_probabilistic = int(prob_0 * prob_1 * n_samples)
        tp_probabilistic = int(prob_1 * prob_1 * n_samples)
        f1_score_probabilistic = tp_probabilistic / (tp_probabilistic + 0.5 * (fp_probabilistic + fn_probabilistic))
        ######################################################
        # ## Find the comparable (in fp) confusion matrix ## #
        pos_nearest = (numpy.abs(fp_curve - fp_probabilistic)).argmin()
        n_times = tp_curve[pos_nearest] / tp_probabilistic
        ######################################################
        print(f"The tested model is '{n_times}' times best than probabilistic bechmark detecting the minority class")
        print("probabilistic_random_model")
        print(f"\t f1 {numpy.round(f1_score_probabilistic, 4)} \t"
              f"tn {tn_probabilistic} \t// fp {fp_probabilistic} \t//fn {fn_probabilistic} \t//tp {tp_probabilistic}")
        print("main_model")
        print(f"\t f1 {numpy.round(f1_curve[pos_nearest], 4)}"
              f"\t tn {tn_curve[pos_nearest]} \t// fp {fp_curve[pos_nearest]}"
              f"\t//fn {fn_curve[pos_nearest]} \t//tp {tp_curve[pos_nearest]}")
    if verbose:
        print(f"Metrics for 'default' working point = {mean_probability}")
        print(f"\t tn   {tn_curve[pos_default_working_point]} \t//fp {fp_curve[pos_default_working_point]}"
              f"\t//fn {fn_curve[pos_default_working_point]} \t//tp {tp_curve[pos_default_working_point]}")
        print(f"\t Ap0: {ap0_curve[pos_default_working_point]} \t //Ap1:{ap1_curve[pos_default_working_point]}")
        print(f"\t f1:         {f1_curve[pos_default_working_point]}")
        print(f"\t Precission: {precission_curve[pos_default_working_point]}")
        print(f"\t Recall:     {recall_curve[pos_default_working_point]}")
        print(f"\t Acc:        {acc_curve[pos_default_working_point]}")
        print(f"Metrics for max_f1 working point = {working_point_best_f1}")
        print(f"Metrics for max_p_f1 working point = {working_point_best_p_f1}")
        print(f"\t tn   {tn_curve[pos_best_f1]} \t//fp {fp_curve[pos_best_f1]}"
              f"\t//fn {fn_curve[pos_best_f1]} \t//tp {tp_curve[pos_best_f1]}")
        print(f"\t Ap0: {ap0_curve[pos_best_f1]} \t //Ap1:{ap1_curve[pos_best_f1]}")
        print(f"\t f1:         {f1_curve[pos_best_f1]}")
        print(f"\t Precission: {precission_curve[pos_best_f1]}")
        print(f"\t Recall:     {recall_curve[pos_best_f1]}")
        print(f"\t Acc:        {acc_curve[pos_best_f1]}")
    if plot:
        matplotlib.pyplot.figure(figsize=(12, 6))
        matplotlib.pyplot.xlabel('working_point')
        matplotlib.pyplot.plot(axis_working_points[pos_default_working_point],
                               f1_curve[pos_default_working_point],
                               '*r',
                               label='default_wp')
        matplotlib.pyplot.plot(axis_working_points[pos_best_f1], f1_curve[pos_best_f1], 'ob', label='max_f1_wp')
        matplotlib.pyplot.plot(axis_working_points[pos_best_p_f1], p_f1_curve[pos_best_p_f1], 'og', label='max_p_f1_wp')
        matplotlib.pyplot.plot(axis_working_points, f1_curve, label='F1 Score')
        matplotlib.pyplot.plot(axis_working_points, p_f1_curve, label='Proba F1 Score')
        matplotlib.pyplot.plot(axis_working_points, p_precision_curve, label='p_precision_curve')
        matplotlib.pyplot.plot(axis_working_points, p_recall_curve, label='p_recall_curve')
        matplotlib.pyplot.plot(axis_working_points[pos_default_working_point],
                               precission_curve[pos_default_working_point],
                               '*r')
        matplotlib.pyplot.plot(axis_working_points[pos_best_f1], precission_curve[pos_best_f1], 'ob')
        matplotlib.pyplot.plot(axis_working_points[pos_best_p_f1], p_precision_curve[pos_best_p_f1], 'og')
        matplotlib.pyplot.plot(axis_working_points, precission_curve, label='Precission')
        matplotlib.pyplot.plot(axis_working_points[pos_default_working_point],
                               recall_curve[pos_default_working_point],
                               '*r')
        matplotlib.pyplot.plot(axis_working_points[pos_best_f1], recall_curve[pos_best_f1], 'ob')
        matplotlib.pyplot.plot(axis_working_points[pos_best_p_f1], p_recall_curve[pos_best_p_f1], 'og')
        matplotlib.pyplot.plot(axis_working_points, recall_curve, label='Recall')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
        matplotlib.pyplot.figure(figsize=(12, 6))
        matplotlib.pyplot.xlabel('working_point')
        matplotlib.pyplot.plot(axis_working_points, p_tn_curve, label='p_tn_curve')
        matplotlib.pyplot.plot(axis_working_points, p_fp_curve, label='p_fp_curve')
        matplotlib.pyplot.plot(axis_working_points, p_fn_curve, label='p_fn_curve')
        matplotlib.pyplot.plot(axis_working_points, p_tp_curve, label='p_tp_curve')
        matplotlib.pyplot.plot(axis_working_points, tn_curve, label='tn_curve')
        matplotlib.pyplot.plot(axis_working_points, fp_curve, label='fp_curve')
        matplotlib.pyplot.plot(axis_working_points, fn_curve, label='fn_curve')
        matplotlib.pyplot.plot(axis_working_points, tp_curve, label='tp_curve')
    dict_metrics_curve = {"tn_curve": tn_curve,
                          "fp_curve": fp_curve,
                          "fn_curve": fn_curve,
                          "tp_curve": tp_curve,
                          "f1_curve": f1_curve,
                          "p_f1_curve": p_f1_curve,
                          "p_precision_curve": p_precision_curve,
                          "p_recall_curve": p_recall_curve,
                          "p_tn_curve": p_tn_curve,
                          "p_fp_curve": p_fp_curve,
                          "p_fn_curve": p_fn_curve,
                          "p_tp_curve": p_tp_curve,
                          "precission_curve": precission_curve,
                          "recall_curve": recall_curve,
                          "acc_curve": acc_curve,
                          "ap1_curve": ap1_curve,
                          "ap0_curve": ap0_curve}
    results = {'working_point_best_f1': working_point_best_f1,
               'working_point_best_p_f1': working_point_best_p_f1,
               'axis_working_points': axis_working_points,
               'dict_metrics_curve': dict_metrics_curve}
    return results


def build_pipeline(input_df, params_dict, monotonic_restr=False, verbose=False):
    # Distinguish between numerical and categorical variables
    categorical_feat = input_df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numerical_feat = input_df.select_dtypes(exclude=["datetime64", "object", "bool", "category"]).columns.tolist()

    if verbose == True:
        print(f"categorical_features: {categorical_feat}")
        print(f"numerical_feat: {numerical_feat}")

    # Create pipeline
    preprocessor = FeatureUnion([
        ('numerical', make_pipeline(Columns(names=numerical_feat), RobustScaler())),
        ('categorical', make_pipeline(Columns(names=categorical_feat), OneHotEncoder(sparse=False)))
    ])

    # First pipeline to get the names of the features
    pipeline = Pipeline([
        ('preprocessor', preprocessor)])
    pipeline.fit(input_df)

    num_feat = pipeline.named_steps["preprocessor"].transformer_list[0][1].named_steps[
        "robustscaler"].get_feature_names_out().tolist()
    cat_feat = pipeline.named_steps["preprocessor"].transformer_list[1][1].named_steps[
        "onehotencoder"].get_feature_names_out().tolist()
    feat = num_feat + cat_feat

    if monotonic_restr:
        positive_monotonic_variables = []
        negative_monotonic_variables = []

    else:
        positive_monotonic_variables = []
        negative_monotonic_variables = []

    non_monotonic_variables = list(set(feat) - set(positive_monotonic_variables) - set(negative_monotonic_variables))

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', monotonic_LGBM(all_variables=feat,
                                      non_monotonic_variables=non_monotonic_variables,
                                      positive_monotonic_variables=positive_monotonic_variables,
                                      negative_monotonic_variables=negative_monotonic_variables))
    ])

    if verbose == True:
        print(f"Num Total model input variables: {len(feat)}")
        print(f"Num positive_monotonic_variables: {len(positive_monotonic_variables)}")
        print(f"Num negative_monotonic_variables: {len(negative_monotonic_variables)}")
        print(f"Num non_monotonic_variables: {len(non_monotonic_variables)}")

    return pipeline
