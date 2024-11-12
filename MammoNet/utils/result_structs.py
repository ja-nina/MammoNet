from typing import List
from pathlib import Path
from pydantic import BaseModel, ConfigDict, PlainSerializer
from typing_extensions import Annotated
import numpy as np


NdArray = Annotated[
    np.ndarray,
    PlainSerializer(lambda x: x.tolist(), return_type=list),
]

PathType = Annotated[
    Path,
    PlainSerializer(lambda x: str(x), return_type=str),
]


class ImageResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: PathType
    label: int
    prediction: int


class Metrics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: NdArray
    classification_report: str


class ResultsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path_model: PathType
    metrics: Metrics
    image_results: List[ImageResult]


# Example usage
if __name__ == "__main__":
    image_result = ImageResult(image_path=Path("path/to/image.png"), label=1, prediction=0)
    metrics = Metrics(
        accuracy=0.9,
        precision=0.8,
        recall=0.7,
        f1_score=0.75,
        confusion_matrix=np.array([[50, 10], [5, 35]]),
        classification_report="classification report",
    )
    results_config = ResultsConfig(path_model=Path("path/to/model.pth"), metrics=metrics, image_results=[image_result])
    results_config.model_dump()
