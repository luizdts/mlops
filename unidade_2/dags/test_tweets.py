import pytest
import wandb
import math
import pandas as pd

# This is global so all tests are collected under the same run

run = wandb.init(project="mlops_tweets_classifying", job_type="data_checks")

@pytest.fixture(scope="session")
def data():
    # Assume 'run' is a global object in your testing environment
    artifact = run.use_artifact('mlops_tweets_classifying/clean_data:latest', type='CleanedData')
    path = artifact.get_path('clean_data.csv')
    clean_data = path.download()
    df = pd.read_csv(clean_data)
    return df

def test_target_labels(data):
    # Ensure that the 'target' column has only 0 and 1 as labels, excluding NaN
    actual_labels = set(data['target'].unique())

    # Check for equality excluding NaN
    assert all(math.isnan(label) or label in {0.0, 1.0} for label in actual_labels)

def test_dataset_size(data):
    # Ensure that the dataset has at least 1000 rows
    assert len(data) >= 1000

def test_final_column_type(data):
    # Ensure that the 'final' column is of type string
    assert data['final'].dtype == 'O'
