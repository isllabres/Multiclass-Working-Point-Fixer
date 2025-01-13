from Multiclass_classification.src.utils.metric_utils import p_fscore
from Multiclass_classification.src.utils.utils import compute_proba_f1, metrics_scanning_working_points_new
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd


def compare_f1_vs_pf1(y_true, y_pred, th=0.5):

    pf1 = p_fscore(y_true, y_pred)
    proba_f1 = compute_proba_f1(pd.DataFrame(y_true), y_pred, working_point=th)[0]
    f1 = f1_score(y_true,
                  np.array(y_pred >= th, dtype=float),
                  average='binary',
                  pos_label=y_true.max().item(),
                  zero_division=1)

    return pf1, proba_f1, f1


if __name__ == '__main__':

    pf1s, proba_f1s, f1s = [], [], []
    n = 1000000

    y_true = np.array(np.random.rand(n) >= 0.90, dtype=float)
    y_pred = np.random.rand(n)

    ib = float((y_true == 0).sum() / (y_true == 1).sum())
    print(f"Total IB: {ib}")

    #y_true = pd.read_csv("../data/07_output/y_train_true.csv").drop('Unnamed: 0', axis=1).values.ravel()
    #y_pred = pd.read_csv("../data/07_output/y_train_proba.csv").drop('Unnamed: 0', axis=1).values.ravel()

    #y_pred = (y_pred-min(y_pred))/(max(y_pred)-min(y_pred))

    for threshold in np.arange(0.0, 1.0, 0.01):
        print(f"Comparing F1 and PF1 with threshold = {threshold}")

        pf1, proba_f1, f1 = compare_f1_vs_pf1(y_true=y_true, y_pred=y_pred, th=threshold)

        pf1s.append(pf1)
        proba_f1s.append(proba_f1)
        f1s.append(f1)

    f1_metrics = pd.DataFrame()
    f1_metrics["pf1"] = pf1s
    f1_metrics["proba_f1s"] = proba_f1s
    f1_metrics["f1s"] = f1s
    f1_metrics.plot()

    metrics_scanning_working_points_new(pred_proba=y_pred,
                                        y_true=pd.DataFrame(y_true),
                                        imbalance_ratio=1.0,
                                        first_working_point=0.0,
                                        last_working_point=0.99,
                                        n_working_points=100,
                                        fixed_best_working_point=None,
                                        fixed_best_working_point_p_f1=None,
                                        plot=True,
                                        verbose=False,
                                        vs_probabilistic_benchmark=False)

    print("END")