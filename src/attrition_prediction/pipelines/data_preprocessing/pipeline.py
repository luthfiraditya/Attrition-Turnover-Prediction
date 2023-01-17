"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_attrition


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
                func=preprocess_attrition,
                inputs="attrition",
                outputs="preprocessed_attrition",
                name="preprocess_attrition_node",
            )])
