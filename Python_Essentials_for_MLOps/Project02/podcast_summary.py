"""
    UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE
    DEPARTAMENTO DE COMPUTAÇAO E AUTOMAÇAO

    DISCENTE: LUIZ HENRIQUE ARAUJO DANTAS
    PROJETO 2 - AIRFLOW PARA PODCASTS
"""

import os
import json
import logging
import requests
import xmltodict

import pendulum
from airflow.decorators import dag, task
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.exceptions import AirflowException

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
PROJECT_FOLDER = "/home/luizdts/airflow"
EPISODE_FOLDER = "/home/luizdts/airflow/episodes"
FRAME_RATE = 16000

logging.basicConfig(level=logging.INFO)  # Configura o nível de log conforme necessário

@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary():
    """
    Cria uma tabela SQLite chamada 'episodes' se ela ainda não existir no banco de dados 'podcasts'.
    A tabela 'episodes' é usada para armazenar informações sobre os episódios de um podcast,
    incluindo link, título, nome do arquivo, data de publicação, descrição e transcrição.

    A função utiliza o operador SqliteOperator para executar uma consulta SQL que cria a tabela
    'episodes' se ela ainda não existir no banco de dados 'podcasts'.
    """
try:
    create_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )
    create_database.execute(context=None)  # Execute a operação de criação da tabela

    logging.info("Tabela 'episodes' criada com sucesso.")
except AirflowException as e_create_table:
    logging.error("Ocorreu um erro ao criar a tabela 'episodes': str(%s)", e_create_table)

    def fetch_podcast_feed():
        """
        Realiza uma solicitação HTTP GET para a URL do podcast especificada em 'PODCAST_URL'
        e analisa o conteúdo XML da resposta em um formato de dicionário Python usando a
        biblioteca xmltodict.
        """
        try:
            data = requests.get(PODCAST_URL, timeout=10)
            data.raise_for_status()  # Lança uma exceção se a resposta HTTP não for bem-sucedida
            return xmltodict.parse(data.text)

        except requests.exceptions.RequestException as re:
            logging.error("Erro ao fazer a solicitação HTTP: str(%s)", re)
            return None


    def print_episode_count(episodes):
        """
        Retorna a quantidade de episódios encontrados
        """
        print(f"Found {len(episodes)} episodes.")

    @task()
    def get_episodes():
        """
        Obtém os episódios de um podcast a partir do feed RSS recuperado usando
        a função 'fetch_podcast_feed'.
        Imprime o número de episódios encontrados e retorna a lista de episódios.
        """
        try:
            feed = fetch_podcast_feed()
            episodes = feed["rss"]["channel"]["item"]
            print_episode_count(episodes)
            return episodes
        except Exception as e_fetch_podcast_feed:
            logging.error("Ocorreu um erro ao obter eps do podcast: str(%s)", e_fetch_podcast_feed)
            return None

    podcast_episodes = get_episodes()
    create_database.set_downstream(podcast_episodes)

    @task()
    def load_episodes(episodes):
        """
        Carrega novos episódios de um podcast na base de dados SQLite 'podcasts'
        se eles não estiverem já armazenados.
        Os novos episódios são identificados pelo link.
        """
        try:
            hook = SqliteHook(sqlite_conn_id="podcasts")
            stored_links = set(hook.get_pandas_df("SELECT link from episodes;")["link"])
            new_episodes = []

            for episode in episodes:
                link = episode["link"]
                if link not in stored_links:
                    filename = os.path.basename(link) + ".mp3"
                    new_episode = {
                        "link": link,
                        "title": episode["title"],
                        "published": episode["pubDate"],
                        "description": episode["description"],
                        "filename": filename
                    }
                    new_episodes.append(new_episode)

            if new_episodes:
                hook.insert_rows(
                    table='episodes',
                    rows=new_episodes,
                    target_fields=["link", "title", "published", "description", "filename"]
                )

            return new_episodes
        except Exception as e_load_episodes:
            logging.error("Ocorreu um erro ao carregar episódios: str(%s)", e_load_episodes)
            return None

    @task()
    def download_episodes(episodes):
        """
        Faz o download de episódios de áudio de um podcast e os armazena localmente
        no diretório especificado.
        Os episódios são identificados pelo link e são baixados apenas se ainda não
        existirem localmente.
        """
        try:
            audio_files = []
            for episode in episodes:
                link = episode["link"]
                filename = get_filename_from_link(link)
                audio_path = os.path.join(EPISODE_FOLDER, filename)

                if not os.path.exists(audio_path):
                    download_audio(episode["enclosure"]["@url"], audio_path)
                audio_files.append({"link": link, "filename": filename})

            return audio_files
        except Exception as e_download_episodes:
            logging.error("Ocorreu um erro ao carregar episódios: str(%s)", e_download_episodes)
            return None

    def get_filename_from_link(link):
        """
        Obtém o nome do arquivo de áudio a partir de um link.
        O nome do arquivo é derivado do último segmento do URL.
        """
        name_end = link.split('/')[-1]
        return f"{name_end}.mp3"

    def download_audio(audio_url, audio_path):
        """
        Faz o download de um arquivo de áudio a partir de uma URL
        e o armazena no caminho especificado localmente.
        """
        print(f"Downloading {audio_path}")
        audio = requests.get(audio_url, timeout=15)
        with open(audio_path, "wb+") as f:
            f.write(audio.content)

    @task()
    def speech_to_text():
        """
        Realiza a transcrição de áudio para texto dos episódios de um podcast
        e atualiza o banco de dados com as transcrições.
        Os episódios a serem transcritos são identificados pelo campo 'transcript'
        ainda não preenchido no banco de dados.
        """
        try:
            hook = SqliteHook(sqlite_conn_id="podcasts")
            untranscribed_episodes = hook.get_pandas_df(
                "SELECT * from episodes WHERE transcript IS NULL;")

            model = Model(model_name="vosk-model-en-us-0.22-lgraph")
            rec = KaldiRecognizer(model, FRAME_RATE)
            rec.SetWords(True)

            for row in untranscribed_episodes.iterrows():
                print(f"Transcribing {row['filename']}")
                filepath = os.path.join(EPISODE_FOLDER, row["filename"])
                transcript = transcribe_audio(filepath)
                update_transcript_in_database(hook, row["link"], transcript)
        except Exception as e_speech :
            logging.error("Ocorreu um erro ao realizar a transcrição: %s", e_speech)

    def transcribe_audio(filepath):
        """
        Realiza a transcrição de áudio de um arquivo de áudio local em partes
        e retorna a transcrição completa.
        """
        mp3 = load_and_prepare_audio(filepath)
        transcript = ""

        step = 20000
        for i in range(0, len(mp3), step):
            print(f"Progress: {i/len(mp3)}")
            segment = mp3[i:i+step]
            text = transcribe_segment(segment)
            transcript += text

        return transcript

    def load_and_prepare_audio(filepath):
        """
        Carrega e prepara o áudio.
        """
        try:
            mp3 = AudioSegment.from_mp3(filepath)
            mp3 = mp3.set_channels(1)
            mp3 = mp3.set_frame_rate(FRAME_RATE)
            return mp3

        except CouldntDecodeError as e:
            logging.error("Ocorreu um erro ao carregar e preparar o áudio: str(%s)", e)
            return None

    def transcribe_segment(rec, segment):
        """
        Realiza a transcrição de áudio de um segmento de áudio usando um objeto
        KaldiRecognizer 'rec' e retorna o texto transcrito.

        """
        rec.AcceptWaveform(segment.raw_data)
        result = rec.Result()
        text = json.loads(result)["text"]
        return text

    def update_transcript_in_database(hook, link, transcript):
        """
        Atualiza o banco de dados 'episodes' com a transcrição de 
        um episódio identificado pelo 'link'.
        Se o episódio já tiver uma transcrição,
        ela será substituída pela nova transcrição fornecida.
        """
        hook.insert_rows(table='episodes', rows=[[link, transcript]],
                          target_fields=["link", "transcript"], replace=True)

SUMMARY = podcast_summary()
