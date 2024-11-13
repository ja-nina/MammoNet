import torch
from tqdm import tqdm
from sklearn import metrics
from pathlib import Path
from MammoNet.models.BaseModel import BaseModel
from MammoNet.utils.result_structs import ImageResult, ResultsConfig, Metrics


class Evaluator:
    def __init__(self, data_loader_val, data_loader_test):
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metric_functions = {
            "accuracy": metrics.accuracy_score,
            "precision": lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average="weighted"),
            "recall": lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average="weighted"),
            "f1_score": lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": metrics.confusion_matrix,
            "classification_report": metrics.classification_report,
            "auc_score": lambda y_true, y_pred: metrics.roc_auc_score(y_true, y_pred, average="weighted"),
        }

    def _get_predictions(self, model: BaseModel, split="test") -> ImageResult:
        model.model.to(self.device)
        model.model.eval()
        predictions = []
        ground_truths = []
        data_loader = self.data_loader_val if split == "val" else self.data_loader_test

        with torch.no_grad():

            for data in tqdm(data_loader, desc=f"Predicting {split} split"):
                inputs, ground_truth = data
                inputs = inputs.to(self.device)
                outputs = model.model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().tolist())
                ground_truths.extend(ground_truth.cpu().tolist())

        image_paths = data_loader.dataset.file_paths

        return image_paths, ground_truths, predictions

    def _compute_metrics(self, y_true, y_pred):
        """
        Evaluate the model using various metrics.
        """
        computed_metrics = Metrics(**{name: func(y_true, y_pred) for name, func in self.metric_functions.items()})
        return computed_metrics

    def evaluate_model(self, model_class: object, model_path: Path):
        """
        Load, predict and evaluate the model.
        """
        model = self._load_model(model_class, model_path)
        image_paths, ground_truths, predictions = self._get_predictions(model)

        # intermediate results
        results = [
            ImageResult(image_path=path, label=label, prediction=pred)
            for path, label, pred in zip(image_paths, ground_truths, predictions)
        ]

        # metric computation
        computed_metrics = self._compute_metrics(ground_truths, predictions)

        results_config = ResultsConfig(path_model=str(model_path), metrics=computed_metrics, image_results=results)

        return results_config

    def _load_model(self, model_class: BaseModel, model_path: Path):
        """
        Load the model from the given path.
        """
        model = model_class()
        model.load_model(model_path, self.device)
        return model
