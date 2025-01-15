import pandas
from loguru import logger


def load_multiclass_working_point_fixer_data(data_path, data_mode: str = 'binary'):

    if data_mode == 'binary':
        
        y_train_true = pandas.read_csv(f"{data_path}/01_raw/binary_y_train_true.csv")
        y_train_pred = pandas.read_csv(f"{data_path}/01_raw/binary_y_train_pred.csv")
        y_test_true = pandas.read_csv(f"{data_path}/01_raw/binary_y_test_true.csv")
        y_test_pred = pandas.read_csv(f"{data_path}/01_raw/binary_y_test_pred.csv")

    elif data_mode == 'multiclass':

        y_train_true = pandas.read_csv(f"{data_path}/01_raw/multiclass_y_train_true.csv")
        y_train_pred = pandas.read_csv(f"{data_path}/01_raw/multiclass_y_train_pred.csv")
        y_test_true = pandas.read_csv(f"{data_path}/01_raw/multiclass_y_test_true.csv")
        y_test_pred = pandas.read_csv(f"{data_path}/01_raw/multiclass_y_test_pred.csv")

    else:
        logger.error("not valid data_mode argument")

        y_train_true = None
        y_train_pred = None
        y_test_true = None
        y_test_pred = None

    return y_train_true, y_train_pred, y_test_true, y_test_pred
