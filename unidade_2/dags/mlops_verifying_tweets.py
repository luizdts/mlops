
import os
import re
import requests
import numpy as np
import wandb
import nltk
import pytest
import logging
import subprocess
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator


from nltk.corpus import stopwords
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import TFAutoModelForSequenceClassification




# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(
    filename="mlops_verifying_tweets.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)





WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# ------ DEFINE TASKS --------------------------------------------------------------------------------------

def login_wandb_run():
    wandb.login(WANDB_API_KEY)



def init_wandb_run(project, job_type, save_code=True):
    return wandb.init(project=project, job_type=job_type, save_code=save_code)

run = init_wandb_run(project='mlops_tweets_classifying', job_type='init')

def fetch_data():
    login_wandb_run()
    ##run = init_wandb_run(project='mlops_tweets_classifying', job_type='data_fetching')

    # Download dataset
    url = 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'
    response = requests.get(url)

    with open('train.csv', 'wb') as file:
        file.write(response.content)

    # Send the raw_data to W&B and store it as an artifact
    logging.info("Sending the raw_data to W&B and storing it as an artifact")
    subprocess.run([
        'wandb', 'artifact', 'put',
        '--name', 'mlops_tweets_classifying/raw_data',
        '--type', 'RawData',
        '--description', 'Real and Fake Disaster-Related Tweets Dataset',
        'train.csv'
    ])


# Explore the data and log relevant information and plots
def data_exploration():
    #run = init_wandb_run(project='mlops_tweets_classifying', job_type='data_exploration')

    # Get the artifact
    artifact = run.use_artifact('mlops_tweets_classifying/raw_data:latest', type='RawData')
    df = pd.read_csv(artifact.file())

    # Log target value counts
    target_value_counts = df['target'].value_counts()
    wandb.log({"Target Value Counts": target_value_counts.to_dict()})

    # Log normalized target value counts
    normalized_target_value_counts = df['target'].value_counts(normalize=True)
    wandb.log({"Normalized Target Value Counts": normalized_target_value_counts.to_dict()})

    # Plot countplot and log the image
    plot_countplot(df)

# Helper function to plot countplot
def plot_countplot(df):
    sns.countplot(x='target', data=df)
    plt.title('Tweet Count by Category')
    plt.savefig('file_category_tweet.png')
    plt.show()
    plt.close()

    # Log the countplot image to wandb
    wandb.log({"File Category Tweet": wandb.Image('file_category_tweet.png')})



# Preprocess data and store the cleaned data as an artifact
def preprocessing_data():
    #run = init_wandb_run(project='mlops_tweets_classifying', job_type='preprocessing')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')

    # Get the artifact
    artifact = run.use_artifact('mlops_tweets_classifying/raw_data:latest', type='RawData')
    df = pd.read_csv(artifact.file())

    df = df.drop(['id','keyword', 'location'], axis=1)

    # Apply text preprocessing
    df = preprocess_text(df)

    # Save the cleaned DataFrame as a CSV file
    df.to_csv('clean_data.csv', index=False)

    # Send the clean_data to W&B and store it as an artifact
    subprocess.run([
        'wandb', 'artifact', 'put',
        '--name', 'mlops_tweets_classifying/clean_data',
        '--type', 'CleanedData',
        '--description', 'Preprocessed data',
        'clean_data.csv'
    ])

# Helper function to preprocess text
def preprocess_text(df):
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    df['text_tokenized'] = df['text'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    df['text_stop'] = df['text_tokenized'].apply(lambda x: [word for word in x if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    df['text_lemmatized'] = df['text_stop'].apply(lambda x: [lemmatizer.lemmatize(word=word, pos='v') for word in x])
    df['final'] = df['text_lemmatized'].str.join(' ')
    return df



def data_check():
    result = pytest.main(["-vv", "."])

    if result != 0:
        raise ValueError("Data checks failed")




# Split data into train and test sets, log information, and store as artifacts
def data_segregation():
    #run = init_wandb_run(project='mlops_tweets_classifying', job_type='data_segregation')

    # Get the clean_data artifact
    artifact = run.use_artifact('mlops_tweets_classifying/clean_data:latest', type='CleanedData')
    path = artifact.get_path('clean_data.csv')
    clean_data = path.download()
    df = pd.read_csv(clean_data)

    # Split the data into training and testing sets
    train_data, test_data = split_data(df)

    # Log the shapes of the training and testing datasets
    wandb.log({'train_data_shape': train_data.shape, 'test_data_shape': test_data.shape})

    # Save split data to CSV files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

       # Create new artifacts for train and test data
    train_artifact = wandb.Artifact(
        name='train_data',
        type='TrainData',
        description='Training data split from cleanData'
    )
    test_artifact = wandb.Artifact(
        name='test_data',
        type='TestData',
        description='Testing data split from cleanData'
    )

    # Add CSV files to the artifacts
    train_artifact.add_file('train_data.csv')
    test_artifact.add_file('test_data.csv')

    # Log the new artifacts to wandb
    run.log_artifact(train_artifact)
    run.log_artifact(test_artifact)


# Helper function to split data
def split_data(df):
    X = df['final']
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    train_data = pd.DataFrame({'final': x_train, 'target': y_train})
    test_data = pd.DataFrame({'final': x_test, 'target': y_test})
    return train_data, test_data


# Helper function to create data artifacts
# def create_data_artifacts(run, name, data_type, description, file_name):
#     artifact = wandb.Artifact(name=name, type=data_type, description=description)
#     artifact.add_file(file_name)
#     run.log_artifact(artifact)



# Train the model and log relevant information and plots
def data_train():
    run = init_wandb_run(project="mlops_tweets_classifying", job_type="train")

    # Get the train and test artifacts
    train_artifact = run.use_artifact('mlops_tweets_classifying/train_data:latest', type='TrainData')
    test_artifact = run.use_artifact('mlops_tweets_classifying/test_data:latest', type='TestData')

    # Load data
    X_train, y_train, X_test, y_test = load_data(train_artifact, test_artifact)

    # Tokenize and create TensorFlow datasets
    train_dataset, test_dataset = preprocess_and_create_datasets(X_train, y_train, X_test, y_test)

    # Define and train the model
    model, history = define_and_train_model(train_dataset, test_dataset)

    # Plot and log training loss and accuracy
    plot_and_log_training_info(history)

# Helper function to load data from artifacts
def load_data(train_artifact, test_artifact):
    train_path = train_artifact.get_entry('train_data.csv').download()
    test_path = test_artifact.get_entry('test_data.csv').download()


    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df['final']
    y_train = train_df['target']
    X_test = test_df['final']
    y_test = test_df['target']

    return X_train, y_train, X_test, y_test

# Helper function to preprocess data and create TensorFlow datasets
def preprocess_and_create_datasets(X_train, y_train, X_test, y_test):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

    train_dataset = create_tf_dataset(train_encodings, y_train)
    test_dataset = create_tf_dataset(test_encodings, y_test)

    return train_dataset, test_dataset

# Helper function to create TensorFlow dataset
def create_tf_dataset(encodings, labels):
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        tf.constant(labels.values, dtype=tf.int32)
    ))
    return dataset.batch(16)

# Helper function to define and train the model
def define_and_train_model(train_dataset, test_dataset):
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(train_dataset, epochs=1, validation_data=test_dataset)

    return model, history

# Helper function to plot and log training loss and accuracy
def plot_and_log_training_info(history):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(history.history["loss"], label="train_loss", linestyle='--')
    ax.plot(history.history["val_loss"], label="val_loss", linestyle='--')
    ax.plot(history.history["accuracy"], label="train_acc")
    ax.plot(history.history["val_accuracy"], label="val_acc")

    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Accuracy")

    ax.legend()
    plt.tight_layout()

    plt.savefig(f"training_loss_acc.png")

    # Log the plot as an image to W&B
    wandb.log({"training_loss_acc": wandb.Image(f"training_loss_acc.png")})


def wandb_finish():
    #run = init_wandb_run(project='mlops_tweets_classifying', job_type='data_fetching')
    run.finish()
    


DEFAULT_ARGS = {
    "owner": "airflow",
    "start_date": datetime(2023, 11, 30),
    "catchup": False,
}

with DAG("mlops_tweets_classifying", default_args=DEFAULT_ARGS, schedule_interval="@daily", catchup=False) as dag:
# ------ FETCH DATA --------------------------------------------------------------------------------------
    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
        op_kwargs={
            "api_key": WANDB_API_KEY
        }
    )


 # ------ EDA --------------------------------------------------------------------------------------
    data_exploration = PythonOperator(
         task_id="data_exploration",
         python_callable=data_exploration
    )

    

# # ------ PREPROCESSING --------------------------------------------------------------------------------------
    preprocessing_data = PythonOperator(
         task_id="preprocessing_data",
         python_callable=preprocessing_data
    )

    

# # ------ DATA CHECK --------------------------------------------------------------------------------------
    data_check = PythonOperator(
         task_id="data_check",
         python_callable=data_check
    )

 # ------ DATA SEGREGATION --------------------------------------------------------------------------------------
    data_segregation = PythonOperator(
         task_id="data_segregation",
         python_callable=data_segregation
    )

    

# # ------ DATA TRAIN --------------------------------------------------------------------------------------
    data_train = PythonOperator(
         task_id="data_train",
         python_callable=data_train
        
    )

    wandb_finish = PythonOperator(
         task_id="wandb_finish",
         python_callable=wandb_finish
        
    )

fetch_data >> data_exploration >> preprocessing_data >> data_check  >> data_segregation >> data_train >> wandb_finish

