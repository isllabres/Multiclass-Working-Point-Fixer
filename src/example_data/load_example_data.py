import pandas
from loguru import logger


def load_example_data(data_mode: str, data_path: str):
    """Load example data based on the specified data mode

    Depending on the data_mode ('binary' or 'multiclass'), this function loads the corresponding training and testing
    true and predicted labels from CSV files located at the specified data_path.

    Args:
        data_mode (str): The mode of the data to load ('binary' or 'multiclass')
        data_path (str): The path to the directory containing the data files

    Returns:
        tuple: A tuple containing the training true labels, training predicted labels, testing true labels, and testing predicted labels
    """
    if data_mode == 'binary':
        y_train_true = pandas.read_csv(f"{data_path}/raw/binary_y_train_true.csv")
        y_train_pred = pandas.read_csv(f"{data_path}/raw/binary_y_train_pred.csv")
        y_test_true = pandas.read_csv(f"{data_path}/raw/binary_y_test_true.csv")
        y_test_pred = pandas.read_csv(f"{data_path}/raw/binary_y_test_pred.csv")

    elif data_mode == 'multiclass':
        y_train_true = pandas.read_csv(f"{data_path}/raw/multiclass_y_train_true.csv")
        y_train_pred = pandas.read_csv(f"{data_path}/raw/multiclass_y_train_pred.csv")
        y_test_true = pandas.read_csv(f"{data_path}/raw/multiclass_y_test_true.csv")
        y_test_pred = pandas.read_csv(f"{data_path}/raw/multiclass_y_test_pred.csv")

    else:
        logger.error("not valid data_mode argument")
        y_train_true = None
        y_train_pred = None
        y_test_true = None
        y_test_pred = None

    return y_train_true, y_train_pred, y_test_true, y_test_pred
