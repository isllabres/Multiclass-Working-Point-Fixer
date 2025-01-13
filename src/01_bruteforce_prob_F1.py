from Multiclass_classification.src.utils.utils import \
    build_pipeline,\
    metrics_scanning_working_points_new
from Multiclass_classification.src.utils.calibrator import calibrate
import pandas as pd
from sklearn.model_selection import train_test_split

def bruteforce_prob_F1(X_train,
                       X_test,
                       y_train,
                       y_test,
                       model_params,
                       ib_train,
                       clf_ir_after_rb,
                       calibrate_probabilities):
    # Build model
    proba_lgbm_model = build_pipeline(X_train, model_params, monotonic_restr=True, verbose=False)

    # Fit model
    proba_lgbm_model.named_steps["classifier"].set_params(model_params)
    proba_lgbm_model.fit(X_train, y_train.values, **{'classifier__ir_after_rebalance': clf_ir_after_rb})

    # Evaluate in train
    print('\nTraining...')
    y_train_proba = proba_lgbm_model.predict_proba(X_train)
    y_test_proba = proba_lgbm_model.predict_proba(X_test)

    if calibrate_probabilities:
        print("Calibration")
        y_train_proba, y_test_proba = calibrate(y_train=y_train.values.ravel(),
                                                          y_train_proba=y_train_proba,
                                                          y_test=y_test.values.ravel(),
                                                          y_test_proba=y_test_proba)

    results_tr = metrics_scanning_working_points_new(pred_proba=y_train_proba,
                                                     y_true=y_train,
                                                     imbalance_ratio=ib_train,
                                                     first_working_point=0.0,
                                                     last_working_point=.999,
                                                     n_working_points=150,
                                                     fixed_best_working_point=None,
                                                     fixed_best_working_point_p_f1=None,
                                                     plot=True,
                                                     verbose=True,
                                                     vs_probabilistic_benchmark=True)

    working_point_best_p_f1_tr = results_tr['working_point_best_p_f1']
    working_point_best_f1_tr = results_tr['working_point_best_f1']
    axis_working_points = results_tr['axis_working_points']
    dict_metrics_curve = results_tr['dict_metrics_curve']

    # Evaluate in test
    print('\nTesting...')
    results_ts = metrics_scanning_working_points_new(pred_proba=y_test_proba,
                                                     y_true=y_test,
                                                     imbalance_ratio=ib_train,
                                                     first_working_point=0.0,
                                                     last_working_point=0.999,
                                                     n_working_points=150,
                                                     fixed_best_working_point=working_point_best_f1_tr,
                                                     fixed_best_working_point_p_f1=working_point_best_p_f1_tr,
                                                     plot=True,
                                                     verbose=True,
                                                     vs_probabilistic_benchmark=True)

    working_point_best_f1_ts = results_ts['working_point_best_f1']
    axis_working_points = results_ts['axis_working_points']
    dict_metrics_curve = results_ts['dict_metrics_curve']

def setup_data(path=None,
               y_cols='fuga_personalizada',
               drop_cols=[]):

    if path is not None:
        data = pd.read_csv(path, sep=',')
        data = data.drop(drop_cols, axis=1)

        data = data.drop([col for col in data.columns if "fecha" in col], axis=1)

        data = data.dropna(axis=1, how='all')
        data = data.dropna(axis=0, how='any')

        X = data.drop(y_cols, axis=1)
        y = data[y_cols]
    else:
        from sklearn.datasets import load_iris

        from imblearn.datasets import make_imbalance

        iris = load_iris()
        X, y = make_imbalance(
            iris.data,
            iris.target,
            sampling_strategy={0: 10, 1: 50, 2: 0},
            random_state=42,
        )

    x_tr, x_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.1, random_state=42)

    ib_total = float((y == 0).sum() / (y == 1).sum())
    ib_train = float((y_tr == 0).sum() / (y_tr == 1).sum())
    ib_test = float((y_tst == 0).sum() / (y_tst == 1).sum())

    print(f"Total IB: {ib_total}, Train IB: {ib_train}, Test IB: {ib_test}")

    return pd.DataFrame(x_tr),\
        pd.DataFrame(x_tst), \
        pd.DataFrame(y_tr), \
        pd.DataFrame(y_tst), \
        ib_train

if __name__ == '__main__':

    #X_train, X_test, y_train, y_test, ib_train = setup_data(path="../data/04_feature/table_train_0.625_volume_ratio_8_target_6_active_min_24_transactions.csv",
    #                                                        y_cols='fuga_personalizada',
    #                                                        drop_cols=['Unnamed: 0',
    #                                                                   'Cliente_Familia',
    #                                                                   'Cliente',
    #                                                                   'Cliente Descripcion'])

    #drop_cols = ['agg_key',
    #             'cliente_id',
    #             'familia_material',
    #             'destino_id',
    #             'pais_id',
    #             'grupo_cliente_id',
    #             'cliente_completo',
    #             'grupo_cliente_completo',
    #             'destino_completo',
    #             'Distrito']
    #X_train, X_test, y_train, y_test, ib_train = setup_data()
    X_train = pd.read_csv("../data/04_feature/glp_espana/glp_x_train.csv").drop('Unnamed: 0', axis=1)
    X_test = pd.read_csv("../data/04_feature/glp_espana/glp_x_test.csv").drop('Unnamed: 0', axis=1)
    y_train = pd.read_csv("../data/04_feature/glp_espana/glp_y_train.csv").drop('Unnamed: 0', axis=1).astype(float)
    y_test = pd.read_csv("../data/04_feature/glp_espana/glp_y_test.csv").drop('Unnamed: 0', axis=1).astype(float)

    ib_train = float((y_train == 0).sum() / (y_train == 1).sum())

    model_params = {"input_size": 2,
                    "learning_rate": 0.005,
                    "momentum": 0.5,
                    "centered": False,
                    "num_epochs": 100,
                    "batch_size": 10,
                    "activate_rebalance_compensation": False}#default a True

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Train imbalance: {ib_train}")

    bruteforce_prob_F1(X_train=X_train,
                       X_test=X_test,
                       y_train=y_train,
                       y_test=y_test,
                       model_params=model_params,
                       ib_train=ib_train,
                       clf_ir_after_rb=5.0,
                       calibrate_probabilities=False)