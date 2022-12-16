import numpy as np
import pandas as pd
from sklearn import base
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Dict

from hamilton.function_modifiers import extract_columns
from hamilton.function_modifiers import parameterize_values, parameterize_sources, parameterize

cols_string = 'PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'
cols = cols_string.strip().split(',')
initial_feature_cols = list(set(cols) - set(['Survived']))
cat_cols = ["sex", "cabin", "embarked"]
combined_cat_cols = ["combined_"+e for e in cat_cols]

def _sanitize_columns(
    df_columns: List[str] # the current column names
) -> List[str]: # sanitized column names
    return [c.strip().replace("/", "_per_").replace(" ", "_").lower() for c in df_columns]

cols_initial_feature_sanitized = _sanitize_columns(initial_feature_cols)

def _label_encoder(
    input_series: pd.Series # series to categorize
) -> preprocessing.LabelEncoder: # sklearn label encoder
    le = preprocessing.LabelEncoder()
    le.fit(input_series)
    return le

@parameterize_sources(
    cabin_encoder = dict(cat='combined_cabin_t'),
    sex_encoder = dict(cat='combined_sex'),
    embarked_encoder = dict(cat='combined_embarked')
)
def cat_encoder(
    cat: pd.Series # series for cat
) -> preprocessing.LabelEncoder: # label encoder for cat
    return _label_encoder(cat)

def encoders(
    cabin_encoder: preprocessing.LabelEncoder,
    sex_encoder: preprocessing.LabelEncoder,
    embarked_encoder: preprocessing.LabelEncoder
) -> Dict:
    return dict(
        cabin_encoder = cabin_encoder,
        sex_encoder = sex_encoder,
        embarked_encoder = embarked_encoder
    )
##### parametric pipeline for both train and test data prep



@parameterize_sources(
    input_data_train = dict(data='df_train'),
    input_data_test = dict(data='df_test'),
    input_data = dict(data='df')
)
def input_data(
    data: pd.DataFrame, # read dataframe
    index_column: str # column to use as unique index
) -> pd.DataFrame: # sanitized dataframe
    data.columns = _sanitize_columns(data.columns)
    return data.set_index(index_column)

def target(
    input_data_train: pd.DataFrame, # input dataframe,
    target_column: str # this is the column that we want from the dataframe
) -> pd.Series: # return series corresponding to target
    return input_data_train[target_column]

@extract_columns(*cols_initial_feature_sanitized)
def features(
    input_data: pd.DataFrame, # input dataframe,
    target_column: str, # this is the column that we want to take out from the dataframe
) -> pd.DataFrame: # return dataframe corresponding to the feature matrix
    print(type(input_data))
    if target_column in input_data.columns:
        return input_data.drop([target_column], axis=1) # new frame
    return input_data

@extract_columns(*combined_cat_cols)
def combined_categoricals(
    input_data_train: pd.DataFrame, # train input
    input_data_test: pd.DataFrame, # test input
    categorical_columns: List[str]
) -> pd.DataFrame: # return combined dataframe of categoricals
    df = pd.concat([input_data_train[categorical_columns], input_data_test[categorical_columns]], axis=0)
    print(df.columns)
    df.columns = ["combined_"+e for e in df.columns]
    return df

def cabin_t(
    cabin: pd.Series # raw cabin info
) -> pd.Series: # transformed cabin info
    return cabin.apply(lambda x: x[:1] if x is not np.nan else np.nan)

def combined_cabin_t(
    combined_cabin: pd.Series # raw cabin info
) -> pd.Series: # transformed cabin info
    return combined_cabin.apply(lambda x: x[:1] if x is not np.nan else np.nan)

def ticket_t(
    ticket: pd.Series # raw ticket number
) -> pd.Series: # transformed ticket number
    return ticket.apply(lambda x: str(x).split()[0])

def family(
    sibsp: pd.Series, # number of siblings
    parch: pd.Series # number of parents/children
) -> pd.Series: # number of people in family
    return sibsp + parch

def _label_transformer(
    fit_le: preprocessing.LabelEncoder, # a fit label encoder
    input_series: pd.Series # series to transform 
) -> pd.Series: # transformed series
    return fit_le.transform(input_series)

# we re-name the encoders here to break the graph

@parameterize_sources(
    cabin_category = dict(cat='cabin_t', cat_encoder='cabinencoder'),
    sex_category = dict(cat='sex', cat_encoder='sexencoder'),
    embarked_category = dict(cat='embarked', cat_encoder='embarkedencoder')
)
def cat_category(
    cat: pd.Series, # cat series
    cat_encoder: preprocessing.LabelEncoder # fit cat labelencoder
) -> pd.Series: # categorized cat
    return _label_transformer(cat_encoder, cat)

def engineered_features(
    pclass: pd.Series, # passenger class extracted column
    age: pd.Series, # age
    fare: pd.Series, # fare
    cabin_category: pd.Series, # categorical cabin
    sex_category: pd.Series, # categorical sex
    embarked_category: pd.Series, # categorical embarked
    family: pd.Series, # constructed family                
) -> pd.DataFrame: # dataframe with dropped columns ready to feed into model
    df = pd.DataFrame({
        'pclass': pclass,
        'age': age,
        'fare': fare,
        'cabin_category': cabin_category,
        'sex_category': sex_category,
        'embarked_category': embarked_category,
        'family': family
    })
    return df

def final_imputed_features(
    engineered_features: pd.DataFrame, # feature matrix with features dropped
) -> pd.DataFrame: # dataframe with imputed columns ready to feed into model
    return engineered_features.fillna(0)