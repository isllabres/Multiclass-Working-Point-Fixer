import pandas
from loguru import logger
import numpy as np
from sklearn.model_selection import train_test_split


def load_example_data(data_mode: str, data_path: str):
    """Load example data based on the specified data mode

    Depending on the data_mode ('binary' or 'multiclass'), this function loads the corresponding training and testing
    true and predicted labels from CSV files located at the specified data_path.

    Args:
        data_mode (str): The mode of the data to load ('binary' or 'multiclass')
        data_path (str): The path to the directory containing the data files

    Returns:
        tuple: A tuple containing the training true labels, training predicted labels, testing true labels, 
        and testing predicted labels
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


def generate_random_multiclass_data(num_samples: int, num_classes: int, val_size: float):
    """Generate random multiclass model predictions probabilities and true labels

    Args:
        num_samples (int): The number of samples to generate
        num_classes (int): The number of classes
        val_size (float): The proportion of the dataset to include in the validation split

    Returns:
        tuple: A tuple containing the training true labels, training predicted probabilities, 
        validation true labels, and validation predicted probabilities
    """
    # Generate random true labels
    y_true = np.random.randint(0, num_classes, size=num_samples)

    # One-hot encode the true labels
    y_true_one_hot = np.eye(num_classes)[y_true]

    # Generate random predicted probabilities
    y_pred_proba = np.random.rand(num_samples, num_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # Normalize to get probabilities

    # Split the data into training and validation sets
    y_train_true, y_val_true, y_train_pred_proba, y_val_pred_proba = train_test_split(
        y_true_one_hot, y_pred_proba, test_size=val_size, random_state=42
    )

    return y_train_true, y_train_pred_proba, y_val_true, y_val_pred_proba
