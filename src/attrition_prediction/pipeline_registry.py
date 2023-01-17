"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from attrition_prediction.pipelines import data_preprocessing as dp
from attrition_prediction.pipelines import data_science as ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_preprocessing_pipeline=dp.create_pipeline()
    data_science_pipeline=ds.create_pipeline()

    return {
        "__default__":data_preprocessing_pipeline+data_science_pipeline,
        "dp":data_preprocessing_pipeline,
        "ds":data_science_pipeline
    }
