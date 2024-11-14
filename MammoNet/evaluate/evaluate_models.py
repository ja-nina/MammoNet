import json
import argparse
import logging
from pathlib import Path

from MammoNet.evaluate.evaluator import Evaluator
from MammoNet.models import VisionTransformer, SimpleCNN, SimpleNN
from MammoNet.dataset.data_handler import DataHandler
from MammoNet.utils.global_variables import RESULTS_DIR


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

models = {
    Path(RESULTS_DIR, "SimpleNNModel_no_aug.pth"): SimpleNN,
    Path(RESULTS_DIR, "SimpleNNModel_with_aug.pth"): SimpleNN,
    Path(RESULTS_DIR, "SimpleNNModel_with_balancing_aug.pth"): SimpleNN,
    Path(RESULTS_DIR, "SimpleCNNModel_no_aug.pth"): SimpleCNN,
    Path(RESULTS_DIR, "SimpleCNNModel_with_aug.pth"): SimpleCNN,
    Path(RESULTS_DIR, "SimpleCNNModel_with_balancing_aug.pth"): SimpleCNN,
    Path(RESULTS_DIR, "VisionTransformerModel_no_aug.pth"): VisionTransformer,
    Path(RESULTS_DIR, "VisionTransformerModel_with_aug.pth"): VisionTransformer,
    Path(RESULTS_DIR, "VisionTransformerModel_with_balancing_aug.pth"): VisionTransformer,
}

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate models and save results.")
    parser.add_argument(
        "--output", type=str, default=Path(RESULTS_DIR, "results_new.json"), help="Path to save the results JSON file."
    )
    args = parser.parse_args()

    data_handler = DataHandler(augment=True, reuse_augmentation=True)
    _, val_dataloader, test_dataloader = data_handler.get_dataset_loaders()
    evaluator = Evaluator(val_dataloader, test_dataloader)

    results_of_evaluator = []
    for  model_path, model_class in models.items():
        results_of_evaluator.append(evaluator.evaluate_model(model_class, model_path))

        results_json = json.dumps([result.model_dump() for result in results_of_evaluator], indent=4)

        file_path = Path(args.output)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(results_json)
