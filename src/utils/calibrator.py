import pandas as pd
from repclass.pricing.hard_optimizers.probability_models.proba_calibrator.monotonic_regresion import Monotonic_calibrator


def calibrate(y_train, y_train_proba, y_test, y_test_proba):

    pd_train = pd.DataFrame({
        'real': y_train * 1.0,
        'modelo': y_train_proba
    })
    pd_train['set'] = 'train'

    pd_test = pd.DataFrame({
        'real': y_test * 1.0,
        'modelo': y_test_proba
    })

    pd_test['set'] = 'test'

    lgbm_calibrator = Monotonic_calibrator()
    lgbm_calibrator.fit(no_calibrated_proba_tr=pd_train[['modelo']],
                        y_tr=pd_train.real,
                        no_calibrated_proba_ts=pd_test[['modelo']],
                        y_ts=pd_test.real,
                        plot_curves=False)

    return lgbm_calibrator.calibrate_proba(pd_train[['modelo']]),\
        lgbm_calibrator.calibrate_proba(pd_test[['modelo']]),