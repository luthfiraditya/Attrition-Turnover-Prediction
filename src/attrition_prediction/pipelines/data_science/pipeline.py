"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data,normalization,train_model,evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="preprocessed_attrition",
                outputs=["x_train", "x_test", "y_train", "y_test"],
                name="sampling_splitting_node",
            ),
            node(
                func=normalization,
                inputs=["x_train","x_test"], 
                outputs=["x_train_norm","x_test_norm"],
                name="normalization_node",
                ),
            node(
                func=train_model,
                inputs=["x_train_norm","y_train"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier","x_test_norm","y_test"],
                outputs=None,
                name="evaluate_model_node"
            )                 
            ])
