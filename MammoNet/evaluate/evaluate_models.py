import json
import argparse
from pathlib import Path

from MammoNet.evaluate.evaluator import Evaluator
from MammoNet.models import VisionTransformer, SimpleCNN
from MammoNet.dataset.data_handler import DataHandler
from MammoNet.utils.global_variables import RESULTS_DIR

models = {
    VisionTransformer: Path(RESULTS_DIR, "VisionTransformerModel.pth"),
    SimpleCNN: Path(RESULTS_DIR, "SimpleCNNModel.pth"),
}

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate models and save results.")
    parser.add_argument(
        "--output", type=str, default=Path(RESULTS_DIR, "results.json"), help="Path to save the results JSON file."
    )
    args = parser.parse_args()

    data_handler = DataHandler(augment=True, reuse_augmentation=True)
    _, val_dataloader, test_dataloader = data_handler.get_dataset_loaders()
    evaluator = Evaluator(val_dataloader, test_dataloader)

    results_of_evaluator = []
    for model_class, model_path in models.items():
        results_of_evaluator.append(evaluator.evaluate_model(model_class, model_path))

    # Serialize the array of Pydantic models to JSON
    results_json = json.dumps([result.model_dump() for result in results_of_evaluator], indent=4)

    # Save the JSON array to a file
    file_path = Path(args.output)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(results_json)
