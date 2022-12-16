from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn import base
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from hamilton.function_modifiers import extract_fields
from hamilton.function_modifiers import parameterize_values, parameterize_sources, parameterize

@extract_fields({'X_train': pd.DataFrame, 'X_valid': pd.DataFrame, 'y_train': pd.Series, 'y_valid': pd.Series})
def train_valid_split_func(
    final_feature_matrix: pd.DataFrame, # feature matrix
    target: pd.Series, # the target or the y
    validation_size_fraction: float, # the validation fraction
    random_state: int, # random state for reproducibility
) -> Dict[str, Union[pd.DataFrame, pd.Series]]: # dictionary of dataframes and Series
    """Function that creates the training & test splits.
    It this then extracted out into constituent components and used downstream.
    """
    # we put stratify=target below, shows we could decompose general
    # logic from some decisions in a module layer
    X_train, X_valid, y_train, y_valid = train_test_split(
        final_feature_matrix, target, test_size=validation_size_fraction, stratify=target
    )
    return {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}


def prefit_clf(
    random_state: int, # get random state from parameters
    max_depth: Union[int, None] # either None or given max_depth hyperparameter
) -> base.ClassifierMixin: # return an unfit Random Forest
    return RandomForestClassifier(max_depth = max_depth, random_state = random_state)

def fit_clf(
    prefit_clf: base.ClassifierMixin, # prefit classifier
    X_train: pd.DataFrame, # transformed features matrix
    y_train: pd.Series, # target column
) -> base.ClassifierMixin:
    """Calls fit on the classifier object; it mutates the classifier and fits it."""
    prefit_clf.fit(X_train, y_train)
    return prefit_clf

@parameterize_sources(
    train_predictions = dict(X='X_train', clf='fit_clf'),
    valid_predictions = dict(X='X_valid', clf='fit_clf'),
    predictions = dict(X='X', clf='clf'),
    chain_predictions = dict(X='final_imputed_features', clf='clf')
)
def predictions(
    clf: base.ClassifierMixin, # already fit classifier
    X:pd.DataFrame, # training or testing dataframe
    t:float = 0.5 # classification probability threshold
) -> Tuple[float, int]: # Probabilities from model, Predictions from model
    y_proba = clf.predict_proba(X)[:, 1]
    y_preds  = 1*(y_proba >= t)
    return y_proba, y_preds