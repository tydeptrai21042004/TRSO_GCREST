from .build import (
    TASK_DEPTH_ESTIMATION,
    TASK_MULTILABEL,
    TASK_OBJECT_DETECTION,
    TASK_REGRESSION,
    TASK_SEMANTIC_SEGMENTATION,
    TASK_SINGLE_LABEL,
    available_datasets,
    build_dataset,
    build_dataset_split,
)

__all__ = [
    "TASK_SINGLE_LABEL", "TASK_MULTILABEL", "TASK_REGRESSION",
    "TASK_SEMANTIC_SEGMENTATION", "TASK_DEPTH_ESTIMATION", "TASK_OBJECT_DETECTION",
    "available_datasets", "build_dataset", "build_dataset_split",
    "DatasetSpec", "dataset_specs", "get_dataset_spec", "registry_as_dicts",
    "PreprocessingProfile", "current_preprocessing_profile",
    "parse_download_mode", "normalize_download_mode", "should_download",
]

from .registry import DatasetSpec, dataset_specs, get_dataset_spec, registry_as_dicts
from .preprocessing import PreprocessingProfile, current_preprocessing_profile
from .download import parse_download_mode, normalize_download_mode, should_download
