"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.4
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Remove unuseful and redundant features

def _remove_unuseful(x:pd.DataFrame)->pd.DataFrame:
    x=x.drop(['EmployeeCount','EmployeeNumber','StandardHours','Over18'],axis=1)
    return x

def _label_encoding(x:pd.DataFrame)->pd.DataFrame:
    categorical_columns = set(x.dtypes[x.dtypes=='object'].index.values)
    #Label Encoding
    le=LabelEncoder()

    for i in categorical_columns:
        x[i]=le.fit_transform(x[i])
    
    return x

def preprocess_attrition(attrition:pd.DataFrame)->pd.DataFrame:
    """Preprocesses the data for attrition.

    Args:
        attrition: Raw data.
    Returns:
        Preprocessed data, remove unusefull and redundant feature then encoding
        categorical variabel
    """
    attrition=_remove_unuseful(attrition)
    attrition=_label_encoding(attrition)
    return attrition