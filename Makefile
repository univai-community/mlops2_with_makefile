# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon


# Need too specify bash in order for conda activate to work.
.ONESHELL:
export PATH := /Users/rahul/miniforge3/envs/ml1-arm64/bin:$(PATH)
SHELL := /bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
#papermill := /Users/rahul/miniforge3/envs/ml1-arm64/bin/papermill
papermill := papermill

# Overall inputs
train_data=data/train.csv
test_data=data/test.csv
data_data=data/test.csv

# Intermediate targets

encoder=intermediate_data/encoder.joblib
train_store=intermediate_data/featurestore_train.pkl
test_store=intermediate_data/featurestore_test.pkl
model=models/rf.joblib

# Final file outputs

infer_test = results/infer1_test.joblib
infer_data = results/infer2_data.joblib


.PHONY: encoder trainstore teststore training infertest inferdata all

all: encoder trainstore teststore training infertest inferdata

encoder: $(encoder)

trainstore: $(train_store)

teststore: $(test_store)

training: $(model)

infertest: $(infer_test)

inferdata: $(infer_data)

$(encoder): $(train_data) $(test_data) encoder.ipynb
	$(papermill) encoder.ipynb output_notebooks/encoder.ipynb

$(train_store): $(encoder) $(train_data) transform.ipynb
	$(papermill) transform.ipynb output_notebooks/transform_train.ipynb

$(test_store): $(encoder) $(test_data) transform.ipynb
	$(papermill) transform.ipynb output_notebooks/transform_test.ipynb -p data $(test_data) -p datatype test

$(model): $(train_data) $(train_store) training.ipynb
	$(papermill) training.ipynb output_notebooks/training.ipynb -p features_file $(train_store) -p training_data $(train_data)

$(infer_test): $(model) $(test_store)
	$(papermill) infer_from_store.ipynb output_notebooks/infer1_test.ipynb -p model_file $(model) -p features_file $(test_store) -p infer_type test

$(infer_data): $(encoder) $(model) $(data_data)
	$(papermill) infer_from_scratch.ipynb output_notebooks/infer2_data.ipynb -p model_file $(model) -p data $(data_data) -p infer_type data
