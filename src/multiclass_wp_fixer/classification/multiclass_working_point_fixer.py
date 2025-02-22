import numpy
import torch
import typing
import pandas

from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
from torch.nn.parameter import Parameter
from torch.optim import RMSprop

from typing import List


class multiclass_Working_Point_Fixer(torch.nn.Module):
    """Multiclass working point fixer.

    This class allows you to find the best working point for maximizing F1 score.
    Similarly to working_point thresholding in a binary problem, in a C-class
    classification problem, a C-dim vector (vector_w) is trained to be the best working point.

    Attributes:
        input_size (int): number of classes. Size of the input array.
        vector_w (torch.nn.parameter.Parameter): Weight vector to be trained. Initialized to one.
        model_params (dict): Dictionary that contains the parameters of the model.
        multiclass_f1_average (str): F1 average type.
        optimizer (torch.optim): optimizer that performs gradient steps.

        train_loss (numpy.array): array of cumulative train loss per epoch.
        val_loss (numpy.array): array of cumulative val loss per epoch.

        train_f1_scores (numpy.array): array of cumulative train f1 score per epoch.
        val_f1_scores (numpy.array): array of cumulative val f1 score per epoch
        train_f1_w_scores (numpy.array): array of cumulative train weighted f1 score per epoch.
        val_f1_w_scores (numpy.array): array of cumulative val weighted f1 score per epoch
        class_f1_w_scores (numpy.array): array of weighted f1 score per class.
        class_f1_scores (numpy.array):  array of f1 score per class.
    """

    def __init__(self, model_params: typing.Dict[str, object] = {}) -> None:
        """This method create and initialize the multiclass_Working_Point_Fixer model.

        Args:
            model_params (dict): Dictionary that contains the parameters of the model.
        """

        torch.nn.Module.__init__(self)

        self.model_params = {
            "input_size": 1,
            "learning_rate": 0.0001,
            "momentum": 0.0,
            "centered": False,
            "num_epochs": 100,
            "multiclass_f1_average": "macro",
            "train_batch_size": 128,
            "val_batch_size": 128,
        }

        for parameter in model_params:
            if parameter in self.model_params:
                self.model_params[parameter] = model_params[parameter]
            else:
                print(f"\n[WARNING]: The parameter '{parameter}' is not supported\n")

        self.input_size = model_params["input_size"]
        self.vector_w = Parameter(
            torch.ones(self.input_size), True
        )  # El True activa el requires_grad
        self.multiclass_f1_average = model_params["multiclass_f1_average"]

        self.optimizer = RMSprop(
            self.parameters(),
            lr=self.model_params["learning_rate"],
            momentum=self.model_params["momentum"],
            centered=self.model_params["centered"],
        )

        self.train_loss = numpy.array([])
        self.val_loss = numpy.array([])

        self.train_f1_scores = numpy.array([])
        self.val_f1_scores = numpy.array([])

        self.train_f1_w_scores = numpy.array([])
        self.val_f1_w_scores = numpy.array([])

        self.class_f1_w_scores = numpy.array([])
        self.class_f1_scores = numpy.array([])

    def set_vector_w(self, tensor_value: torch.Tensor) -> None:
        """Set the value of vector_w

        Args:
            tensor_value: a tensor value of the same dimensions as self.vector_w

        Returns:
            self.vector_w with the new value set.
        """

        self.vector_w = torch.nn.parameter.Parameter(tensor_value, True)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Class forward method.

        Performs an Element-wise multiplication between the input data
        and self.vector_w.

        Args:
            x_input: tensor with the batch data.

        Returns:
            model output.
        """

        o = self.vector_w * x_input
        o = torch.nn.functional.softmax(o, dim=1)

        return o

    @staticmethod
    def soft_p_f1_loss(
        target: torch.Tensor, model_output: torch.Tensor
    ) -> torch.Tensor:
        """Computes the quadratic soft F1 loss.

        The soft F1 or probabilistic F1 is differentiable, so you can turn
        it into a loss and compute the quadratic loss. This loss moves inside
        a 0 to 1 range.

        Args:
            target: true probabilities.
            model_output: predicted probabilities.

        Returns:
            Quadratic macro soft F1 loss.
        """

        epsilon = 1e-7

        p_tp = (target * model_output).sum(axis=0).to(torch.float32)
        p_fp = ((1.0 - target) * model_output).sum(axis=0).to(torch.float32)
        p_fn = (target * (1.0 - model_output)).sum(axis=0).to(torch.float32)

        p_precision = p_tp / (p_tp + p_fp + epsilon)
        p_recall = p_tp / (p_tp + p_fn + epsilon)

        p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)
        p_f1 = torch.mean(p_f1)  # Mean to compute the F1 score "average='macro'"

        return (1 - p_f1) ** 2

    def setup_dataloader(
        self,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        shuffle: bool = False,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Returns the pytorch Dataloader for train and validation partition.

        Args:
            train_dataset: TensorDataset with the train partition.
            val_dataset: TensorDataset with the validation partition.
            shuffle: whether to perform or not data shuffling

        Returns:
            train_loader
            val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.model_params["train_batch_size"],
            shuffle=shuffle,
            num_workers=0,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.model_params["val_batch_size"],
            shuffle=shuffle,
            num_workers=0,
            drop_last=True,
        )

        return train_loader, val_loader

    def compute_scores(
        self,
        stage: str,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        """Calculates the original F1 and the achieved F1. It also
        extends the F1 and F1_w arrays.

        Args:
            stage: string indicating
            targets: Tensor with the target values.
            inputs: Tensor with the input values.
            outputs: Tensor with the output values.
        """

        if not stage:
            f1 = f1_score(targets, inputs, average=None)
            f1_w = f1_score(targets, outputs, average=None)
        elif self.input_size == 2:
            f1 = f1_score(targets, inputs, average="binary")
            f1_w = f1_score(targets, outputs, average="binary")
        else:
            f1 = f1_score(targets, inputs, average=self.multiclass_f1_average)
            f1_w = f1_score(targets, outputs, average=self.multiclass_f1_average)

        if stage == "train":
            self.train_f1_scores = numpy.append(self.train_f1_scores, f1)
            self.train_f1_w_scores = numpy.append(self.train_f1_w_scores, f1_w)
        elif stage == "val":
            self.val_f1_scores = numpy.append(self.val_f1_scores, f1)
            self.val_f1_w_scores = numpy.append(self.val_f1_w_scores, f1_w)
        else:
            self.class_f1_scores = numpy.append(self.class_f1_scores, f1)
            self.class_f1_w_scores = numpy.append(self.class_f1_w_scores, f1_w)

    def compute_imbalance(
        self, train_partition: pandas.DataFrame, test_partition: pandas.DataFrame
    ) -> None:
        """Calculate imbalance per class.

        Args:
            train_partition: pandas.DataFrame containing y_train true labels
            test_partition: pandas.DataFrame containing y_test true labels
        """

        for k, partition in {"train": train_partition, "test": test_partition}.items():

            print(f"\n{k} partition imbalance:\n")

            for j, c in enumerate(partition.columns):
                y = partition.iloc[:, j]
                print(f"{c} = {float((y == 0).sum() / (y == 1).sum()):.3f}")

    def fit(
        self,
        x_tr: numpy.ndarray,
        y_tr: numpy.ndarray,
        x_tst: numpy.ndarray,
        y_tst: numpy.ndarray,
        verbose: bool = False,
        plot: bool = False,
    ) -> None:
        """Fit method.

        Args:
            x_tr: numpy array with the train input data.
            y_tr: numpy array with the train expected output.
            x_tst: numpy array with the test input data.
            y_tst: numpy array with the test expected output.
            plot: boolean indicating whether to plot or not train-val losses
            and F1 vs F1w scores.
        """
        train_loader, val_loader = self.setup_dataloader(
            train_dataset=TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
            val_dataset=TensorDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst)),
            shuffle=False,
        )

        with torch.no_grad():
            self.compute_scores(
                stage="train",
                targets=y_tr.argmax(axis=1),
                inputs=x_tr.argmax(axis=1),
                outputs=self.forward(x_input=torch.from_numpy(x_tr)).argmax(axis=1),
            )
            self.compute_scores(
                stage="val",
                targets=y_tst.argmax(axis=1),
                inputs=x_tst.argmax(axis=1),
                outputs=self.forward(x_input=torch.from_numpy(x_tst)).argmax(axis=1),
            )

        for index_epoch in range(self.model_params["num_epochs"]):

            epoch_train_loss = numpy.array([])
            epoch_val_loss = numpy.array([])

            train_targets = numpy.array([])
            train_inputs_hard = numpy.array([])
            train_model_output_hard = numpy.array([])

            val_targets = numpy.array([])
            val_inputs_hard = numpy.array([])
            val_model_output_hard = numpy.array([])

            for index_batch, data in enumerate(train_loader, 0):
                inputs, target = data

                self.optimizer.zero_grad()
                model_output = self.forward(x_input=inputs)
                loss = self.soft_p_f1_loss(
                    target=target.double(), model_output=model_output.double()
                )

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    epoch_train_loss = numpy.append(
                        epoch_train_loss, loss.detach().numpy().item()
                    )
                    train_targets = numpy.append(
                        train_targets, target.numpy().argmax(axis=1)
                    )
                    train_inputs_hard = numpy.append(
                        train_inputs_hard, inputs.numpy().argmax(axis=1)
                    )
                    train_model_output_hard = numpy.append(
                        train_model_output_hard, model_output.numpy().argmax(axis=1)
                    )

            with torch.no_grad():
                self.train_loss = numpy.append(self.train_loss, epoch_train_loss.mean())
                self.compute_scores(
                    stage="train",
                    targets=train_targets,
                    inputs=train_inputs_hard,
                    outputs=train_model_output_hard,
                )

                self.eval()
                for index_batch, data in enumerate(val_loader, 0):
                    inputs, target = data
                    self.optimizer.zero_grad()

                    validation_out = self.forward(inputs)
                    validation_loss = self.soft_p_f1_loss(
                        target=target.double(), model_output=validation_out.double()
                    )

                    epoch_val_loss = numpy.append(
                        epoch_val_loss, validation_loss.mean()
                    )
                    val_targets = numpy.append(
                        val_targets, target.numpy().argmax(axis=1)
                    )
                    val_inputs_hard = numpy.append(
                        val_inputs_hard, inputs.numpy().argmax(axis=1)
                    )
                    val_model_output_hard = numpy.append(
                        val_model_output_hard, validation_out.numpy().argmax(axis=1)
                    )

                self.val_loss = numpy.append(self.val_loss, epoch_val_loss.mean())
                self.compute_scores(
                    stage="val",
                    targets=val_targets,
                    inputs=val_inputs_hard,
                    outputs=val_model_output_hard,
                )
                self.train()

            if verbose:
                print(
                    f"Epoch {index_epoch:2}/{self.model_params['num_epochs']}. "
                    f"Train Loss: {epoch_train_loss.mean():.5f}. Val Loss: {epoch_val_loss.mean():.5f} "
                    f"Train F1: {self.train_f1_scores[-1]:.5f}. Val F1: {self.val_f1_scores[-1]:.5f}. "
                    f"Train F1w: {self.train_f1_w_scores[-1]:.5f}. Val F1w: {self.val_f1_w_scores[-1]:.5f}"
                )

        if plot:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.train_loss))),
                    y=self.train_loss,
                    mode="lines",
                    name="Train loss",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.val_loss))),
                    y=self.val_loss,
                    mode="lines",
                    name="Val loss",
                )
            )
            fig.update_layout(
                title="Train and Val losses", xaxis_title="Epochs", yaxis_title="Loss"
            )
            fig.show()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.train_f1_scores))),
                    y=self.train_f1_scores,
                    mode="lines",
                    name="Train F1",
                    line=dict(color="blue"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.val_f1_scores))),
                    y=self.val_f1_scores,
                    mode="lines",
                    name="Val F1",
                    line=dict(color="green"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.train_f1_w_scores))),
                    y=self.train_f1_w_scores,
                    mode="lines",
                    name="Train F1w",
                    line=dict(color="blue", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.val_f1_w_scores))),
                    y=self.val_f1_w_scores,
                    mode="lines",
                    name="Val F1w",
                    line=dict(color="green", dash="dash"),
                )
            )
            fig.update_layout(
                title="Train and Val F1 and F1w",
                xaxis_title="Epochs",
                yaxis_title="Score",
            )
            fig.show()

    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        """Predict method.

        Args:
            x: numpy array with the input data.

        Returns:
            numpy array with the predicted output.
        """
        with torch.no_grad():
            x_tensor = torch.from_numpy(x)
            output = self.forward(x_tensor)
            return output.numpy()

    def plot_class_f1_scores(
        self, x: numpy.array, y: numpy.array, c_names: List = None, plot_title: str = ""
    ) -> None:
        """Plot F1 vs F1w scores per label

        Args:
            x: numpy array with the model input.
            y: numpy array with the expected output.
            c_names: List of class names.
        """

        classes = range(0, y.shape[1])
        colors = ["blue", "green"]

        if c_names is None:
            c_names = classes

        self.compute_scores(
            stage=None,
            targets=y.argmax(axis=1),
            inputs=x.argmax(axis=1),
            outputs=self.forward(x_input=torch.from_numpy(x)).argmax(axis=1),
        )

        import plotly.graph_objects as go

        fig = go.Figure()
        for i, class_i in enumerate(classes):
            fig.add_trace(
                go.Bar(
                    x=[c_names[i]],
                    y=[self.class_f1_scores[i]],
                    name="F1",
                    marker_color=colors[0],
                    width=0.4,
                    offset=-0.2,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[c_names[i]],
                    y=[self.class_f1_w_scores[i]],
                    name="F1w",
                    marker_color=colors[1],
                    width=0.4,
                    offset=0.2,
                )
            )

        fig.update_layout(
            title=plot_title,
            xaxis_title="Classes",
            yaxis_title="F1",
            barmode="group",
            showlegend=False,
        )
        fig.show()

    def plot_weights(self, c_names: List = None, normalized: str = False) -> None:
        """Plot model weights. If normalized, the weights are normalized to sum up to the number of classes.
        With the normalization, the weights can be interpreted as follows:
            1. If the weight is 1, the class significance is the same as at the beginning.
            2. If the weight is greater than 1, the class significance is increased.
            3. If the weight is lower than 1, the class significance is decreased.

        Args:
            c_names: list of class names
            normalized: whether to normalize the weights for plotting.
        """

        w = self.vector_w.detach().numpy()
        num_classes = w.shape[0]

        if normalized:
            w = (w / w.sum()) * num_classes

        classes = range(0, num_classes)

        if c_names is None:
            c_names = classes

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=c_names,
                y=w,
                mode="markers",
                name="weights",
                marker=dict(size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=c_names,
                y=[1] * num_classes,
                mode="lines",
                name="reference",
                line=dict(dash="dash", color="red"),
            )
        )

        fig.update_layout(
            title=f"Normalized weights (sums to {num_classes})",
            xaxis_title="Classes",
            yaxis_title="Weights",
            xaxis=dict(tickangle=90),
            showlegend=True,
        )
        fig.show()
