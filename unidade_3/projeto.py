from zenml import pipeline, step
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List

import gradio as gr


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn.naive_bayes import MultinomialNB
import joblib


# Configuração inicial do logging
# Com level logging.INFO, também é englobado o level logging.ERROR
logging.basicConfig(
    filename="mlops_verifying_tweets.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)


@step
def fecth() -> pd.DataFrame:
   try:
      dataset = pd.read_csv("dataset.csv")
      logging.info("Dataset carregado com sucesso!".upper())
      logging.info(dataset.head())
      return dataset
   except FileNotFoundError as e:
        logging.error(f"Erro ao carregar o dataset: {e}")
        raise
   except Exception as e:
        logging.error(f"Erro desconhecido ao carregar o dataset: {e}")
        raise

@step
def data_exploration(df: pd.DataFrame) -> None:
    try:
        # Verifique se o DataFrame não está vazio
        if df.empty:
            logging.warning("O DataFrame está vazio. Nenhuma análise será realizada.")
            return

        logging.info("Estatísticas Descritivas:".upper())
        logging.info(df.describe())

        logging.info("Contagem de Valores Nulos:".upper())
        logging.info(df.isnull().sum())

        # Verifique se a coluna 'Language' existe antes de acessá-la
        if 'Language' in df.columns:
            logging.info("Chaves únicas na coluna Language:")
            logging.info(df['Language'].unique())
        else:
            logging.warning("A coluna 'Language' não existe no DataFrame.")

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a exploração de dados: {str(e)}")


@step
def pre_process_dados(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Iniciando processamento nos dados.".upper())

        # Verificar se o DataFrame não está vazio
        if df.empty:
            logging.warning("O DataFrame está vazio. Nenhuma operação será realizada.")
            return df

        # Preencher valores nulos na coluna 'Language'
        df['Language'].fillna('', inplace=True)

        # Identificar as linhas em que o texto é uma foto
        mask_photo = df['Language'].astype(str).str.startswith("[Photo")

        logging.info("Removendo fotos da coluna Language.".upper())
        # Remover as linhas em que o texto é uma foto
        df = df[~mask_photo]

        # Certificar-se de lidar com valores NaN adequadamente se houver
        logging.info("Certificar-se de lidar com valores NaN adequadamente se houver.".upper())
        df.dropna(subset=['Language'], inplace=True)

        # Verificar se as linhas foram removidas
        logging.info("Verificar se as linhas foram removidas.".upper())
        logging.info(df['Language'].unique())

        # Identificar as linhas em que o texto é um vídeo
        logging.info("Identificar as linhas em que o texto é um vídeo.".upper())
        mask_video = df['Language'].astype(str).str.startswith("[Video")

        # Remover as linhas em que o texto é um vídeo
        df = df[~mask_video]

        # Certificar-se de lidar com valores NaN adequadamente se houver
        logging.info("Certificar-se de lidar com valores NaN adequadamente se houver.".upper())
        df.dropna(subset=['Language'], inplace=True)

        # Verificar se as linhas foram removidas
        logging.info("Verificar se as linhas foram removidas.".upper())
        logging.info(df['Language'].unique())

        return df

    except Exception as e:
        logging.error(f"Ocorreu um erro durante o pré-processamento dos dados: {str(e)}")
        # Se ocorrer um erro, retorne o DataFrame original
        return df




def clean_text(text:str) -> str:
    # Convert to lowercase
    text = str(text).lower()

    # Remove square brackets and contents inside
    text = re.sub(r'\[.*?\]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # Remove punctuation
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)

    # Remove newline characters
    text = re.sub(r'\n', '', text)

    # Remove words containing digits
    text = re.sub(r'\w*\d\w*', '', text)

    return text

def remove_stopwords(text:str) -> str:
    if pd.notnull(text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    return text
def lemmatize_text(text:str)->str:
    if pd.notnull(text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        lemmatized_text = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_text)
    return text

@step
def pre_process_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Iniciando o download dos recursos nltk.".upper())
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')

        logging.info("download dos recursos nltk completo.".upper())

        logging.info("Limpeza da coluna Text".upper())

        # Certificar-se de que a coluna 'Text' existe no DataFrame
        if 'Text' not in df.columns:
            logging.warning("A coluna 'Text' não existe no DataFrame. Nenhuma operação será realizada.")
            return df

        df['Text'] = df['Text'].apply(lambda x: clean_text(x))
        logging.info("Head()")
        logging.info(df.head())

        logging.info("Limpeza da coluna Language.".upper())

        # Certificar-se de que a coluna 'Language' existe no DataFrame
        if 'Language' not in df.columns:
            logging.warning("A coluna 'Language' não existe no DataFrame. Nenhuma operação será realizada.")
            return df

        df['Language'] = df['Language'].apply(lambda x: clean_text(x))

        logging.info("Head()")
        logging.info(df.head())

        logging.info("Valores únicos na coluna Language.".upper())
        logging.info(df['Language'].unique())

        logging.info("Remove stopwords na coluna Text.".upper())
        # Certificar-se de que a coluna 'Text' existe no DataFrame
        if 'Text' not in df.columns:
            logging.warning("A coluna 'Text' não existe no DataFrame. Nenhuma operação será realizada.")
            return df

        df['Text'] = df['Text'].apply(lambda x: remove_stopwords(x))
        logging.info(df.head())

        logging.info("Lemmatize_text na coluna Text.".upper())
        # Certificar-se de que a coluna 'Text' existe no DataFrame
        if 'Text' not in df.columns:
            logging.warning("A coluna 'Text' não existe no DataFrame. Nenhuma operação será realizada.")
            return df

        df['Text'] = df['Text'].apply(lambda x: lemmatize_text(x))
        logging.info(df.head())

        return df

    except Exception as e:
        logging.error(f"Ocorreu um erro durante o pré-processamento de texto: {str(e)}")
        # Se ocorrer um erro, retorne o DataFrame original
        return df



   

@step
def data_train(df: pd.DataFrame) -> List[Pipeline]:
    try:
        # Certificar-se de que as colunas necessárias existem no DataFrame
        required_columns = {'Text', 'Language', 'Label'}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            logging.error(f"As colunas obrigatórias {missing_columns} não existem no DataFrame.")
            return []

        # Separar
        tweets = df['Text']
        language_labels = df['Language']
        sentiment_labels = df['Label']

        logging.info("train_test_split.".upper())
        tweets_train, tweets_test, lang_labels_train, lang_labels_test, sent_labels_train, sent_labels_test = train_test_split(
            tweets, language_labels, sentiment_labels, test_size=0.2, random_state=42
        )

        logging.info("Classificação de language.".upper())
        # Classificação de language
        language_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC())
        ])

        logging.info("Treinar o modelo de classificação de language.".upper())
        # Treinar o modelo de classificação de language
        language_pipeline.fit(tweets_train, lang_labels_train)

        logging.info("Salvar o modelo language_model.".upper())
        joblib.dump(language_pipeline, 'language_model.joblib')

        test_data_language = pd.DataFrame({
            'text': tweets_test,
            'label': lang_labels_test,
        })

        test_data_sentiment = pd.DataFrame({
            'text': tweets_test,
            'label': sent_labels_test
        })

        logging.info("Salvar test_data_language.csv.".upper())
        test_data_language.to_csv('test_data_language.csv', index=False)

        logging.info("Salvar test_data_sentiment.csv.".upper())
        test_data_sentiment.to_csv('test_data_sentiment.csv', index=False)

        logging.info("Definir o pipeline para classificação de sentimentos.".upper())
        # Definir o pipeline para classificação de sentimentos
        sentiment_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])

        logging.info("Treinar o modelo de classificação de sentimentos.".upper())
        # Treinar o modelo de classificação de sentimentos
        sentiment_pipeline.fit(tweets_train, sent_labels_train)

        logging.info("Salvar o modelo sentiment_model.".upper())
        joblib.dump(sentiment_pipeline, 'sentiment_model.joblib')

        return [language_pipeline, sentiment_pipeline]

    except Exception as e:
        logging.error(f"Ocorreu um erro durante o treinamento dos modelos: {str(e)}")
        return []



@step
def predict(models: List[Pipeline]) -> None:
    try:
        # Verificar se há modelos disponíveis
        if len(models) != 2 or not all(isinstance(model, Pipeline) for model in models):
            logging.error("A lista de modelos não está no formato esperado.")
            return

        language_pipeline, sentiment_pipeline = models

        # Prever o idioma de um tweet
        tweet_language = "Las lágrimas caían silenciosas como la lluvia en su corazón"
        logging.info("Tweet para idioma: %s", tweet_language.upper())
        predicted_language = language_pipeline.predict([tweet_language])[0]
        logging.info("Idioma previsto: %s", predicted_language)

        # Prever o sentimento de um tweet
        tweet_sentiment = "iluminando el día con alegría y esperanza"
        logging.info("Tweet para sentimento: %s", tweet_sentiment.upper())
        predicted_sentiment = sentiment_pipeline.predict([tweet_sentiment])[0]
        logging.info("Sentimento previsto: %s", predicted_sentiment)

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a previsão: {str(e)}")



@step
def accuracy_scores(models: List[Pipeline]) -> None:

   try:
      # Avaliar o modelo de identificação de idioma
      language_pipeline, sentiment_pipeline = models

      logging.info("load test_data_language.csv".upper())
      test_data_language = pd.read_csv("test_data_language.csv")

      logging.info("load test_data_sentiment.csv".upper())
      test_data_sentiment = pd.read_csv("test_data_sentiment.csv")

      # Verificar as dimensões dos dados
      logging.info(f"Shape of test_data_language: {test_data_language.shape}".upper())
      logging.info(f"Shape of test_data_sentiment: {test_data_sentiment.shape}".upper())

      # Handle missing values in text data
      test_data_language['text'] = test_data_language['text'].fillna('')  # Fill NaN values with an empty string
      test_data_language['label'] = test_data_language['label'].astype(str)

      logging.info(test_data_language['label'].isnull().sum())


      lang_predictions = language_pipeline.predict(test_data_language["text"])
      lang_report = classification_report(test_data_language['label'], lang_predictions, zero_division=1)
      logging.info("Language identification report:".upper())
      logging.info(lang_report)


      # Avaliar o modelo de identificação de sentimento
      test_data_sentiment['text'] = test_data_sentiment['text'].fillna('')  # Fill NaN values with an empty string

      sent_predictions = sentiment_pipeline.predict(test_data_sentiment['text'])
      sent_report = classification_report(test_data_sentiment['label'], sent_predictions)
      logging.info("Sentiment classification report:".upper())
      logging.info(sent_report)



      logging.info("função Gradio:".upper())
      def show_interface_gradio(input):
         language = language_pipeline.predict([input])[0]
         sentiment = sentiment_pipeline.predict([input])[0]
         output = f'language: {language}, sentiment: {sentiment}'
         logging.info(f'output = language: {language}, sentiment: {sentiment}'.upper())
         return output
      

      # Crie a interface Gradio
      iface = gr.Interface(fn=show_interface_gradio, inputs='textbox', outputs='textbox', live=False)

      # Inicie a interface Gradio
      iface.launch()
      iface.close()
   except Exception as e:
        logging.error(f"Ocorreu um erro durante a avaliação ou no Gradio.: {str(e)}")



@pipeline
def my_pipeline():
  df = fecth()
  data_exploration(df)
  df = pre_process_dados(df)
  df = pre_process_text(df)
  models = data_train(df)
  predict(models)
  accuracy_scores(models)

my_pipeline()


##  zenml up --blocking