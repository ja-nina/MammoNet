from typing import List
from pathlib import Path
from pydantic import BaseModel, ConfigDict, PlainSerializer, BeforeValidator
from typing_extensions import Annotated
import numpy as np


def cast_to_ndarray(value):
    if isinstance(value, list):
        return np.array(value)
    return value

NdArray = Annotated[
    np.ndarray,
    BeforeValidator(cast_to_ndarray),
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
    auc_score: float


class ResultsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path_model: PathType
    metrics: Metrics
    image_results: List[ImageResult]
