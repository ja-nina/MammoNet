from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer
from typing import List, Any
from typing_extensions import Annotated
from pathlib import Path
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
