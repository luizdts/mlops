#-*- coding: utf-8 -*-
"""
    UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE
    DEPARTAMENTO DE COMPUTAÇAO E AUTOMAÇAO

    DISCENTE: LUIZ HENRIQUE ARAUJO DANTAS
    PROJETO 1 - RECOMENDAÇÃO DE FILMES
"""

import re
import logging
import pandas as pd
import ipywidgets as widgets
import numpy as np

from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download datasets: https://files.grouplens.org/datasets/movielens/ml-25m.zip

logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename='error.log',
                    level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    logging.info("Iniciando leitura do arquivo CSV 'movies.csv'.")
    movies = pd.read_csv("movies.csv")
    logging.info("Leitura do arquivo CSV 'movies.csv' concluída com sucesso.")
except Exception as e:
    logging.error("Erro ao ler o arquivo CSV 'movies.csv': %s", e)
    print("Ocorreu um erro ao ler o arquivo CSV 'movies.csv'.")

def clean_title(title):
    """
    Remove caracteres especiais e espaços em branco extras de um título.

    Esta função recebe um título como entrada e remove todos os caracteres
    especiais (exceto letras e números) e espaços em branco extras do título,
    deixando-o limpo e pronto para uso em outras operações.

    Args:
        title (str): O título a ser limpo.

    Returns:
        str: O título limpo, sem caracteres especiais e espaços extras.

    Exemplo:
        Para limpar um título como "Toy Story 1", você pode
        chamar a função da seguinte maneira:

        >>> titulo_limpo = clean_title("Toy Story")
        >>> print(titulo_limpo)
        "ToyStory"
    """
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    title = title.replace(" ", "")
    return title

movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    """
    Realiza uma pesquisa de filmes com base em um título.

    Esta função recebe um título como entrada, limpa o título para remoção de caracteres
    especiais e espaços em branco extras, e em seguida, utiliza uma técnica de similaridade
    para encontrar os filmes mais semelhantes ao título fornecido. Os resultados são
    classificados com base na similaridade e retornados como um DataFrame contendo informações
    sobre os filmes correspondentes.

    Args:
        title (str): O título a ser usado como critério de pesquisa.

    Returns:
        DataFrame: Um DataFrame contendo informações sobre os filmes mais similares ao título
        fornecido, classificados por similaridade decrescente.

    Exemplo:
        Para buscar filmes com base no título "Monsters Inc", você pode chamar a função
        da seguinte maneira:

        >>> result = search("Monsters Inc")
        >>> print(result)
    """
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]

    return results

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

movie_list = widgets.Output()

display(movie_input, movie_list)

def on_type_title(data):
    """
    Exibe uma lista de filmes com base no título digitado pelo usuário.

    Esta função é projetada para ser usada em um ambiente interativo onde os usuários
    digitam um título de filme. Quando o título é digitado (com pelo menos 6 caracteres),
    a função realiza uma pesquisa pelo título e exibe uma lista de filmes correspondentes
    com base na pesquisa.

    Args:
        data (dict): Um dicionário que geralmente contém o título digitado pelo usuário
        como o valor da chave "new".

    Returns:
        None: A função não retorna um valor diretamente, mas exibe a lista de filmes
        correspondentes no ambiente interativo.

    Exemplo:
        Esta função é geralmente usada em um ambiente interativo onde os usuários digitam
        um título de filme. Quando um título com pelo menos 6 caracteres é digitado, a função
        realiza uma pesquisa e exibe uma lista de filmes correspondentes. O exemplo de uso é
        interativo e pode variar dependendo da implementação específica.
    """
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type_title, names='value')

display(movie_input, movie_list)

MOVIE_ID = 89745

#def find_similar_movies(movie_id):
movie = movies[movies["movieId"] == MOVIE_ID]

try:
    logging.info("Iniciando leitura do arquivo CSV 'ratings.csv'.")
    ratings = pd.read_csv("ratings.csv")
    logging.info("Leitura do arquivo CSV 'ratings.csv' realizada com sucesso.")
except Exception as e:
    logging.error("Erro ao ler o arquivo CSV: %s", e)
    print("Ocorreu um erro ao ler o arquivo CSV 'ratings.csv'.")

# Filtra as classificações dos usuários que classificaram o filme com nota > 4
similar_user_ratings = ratings[(ratings["movieId"] == MOVIE_ID) &
                                (ratings["rating"] > 4)]

# Obtém os IDs dos usuários similares
similar_users = similar_user_ratings["userId"].unique()

# Filtra as classificações dos usuários similares com nota > 4
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                             (ratings["rating"] > 4)]["movieId"]

# Calcula a contagem das recomendações de usuários similares
similar_user_recs_count = similar_user_recs.value_counts() / len(similar_users)

# Filtra as recomendações que são mais comuns do que 10% entre os usuários similares
similar_user_recs_count = similar_user_recs_count[similar_user_recs_count > 0.10]

# Filtra todas as classificações dos usuários
all_users = ratings[(ratings["movieId"].isin(similar_user_recs_count.index)) &
                     (ratings["rating"] > 4)]

# Calcula a contagem das recomendações de todos os usuários
all_user_recs_count = all_users["movieId"].value_counts() / len(all_users["userId"].
                                                                unique())

# Cria um DataFrame com as contagens de recomendações similares e de todos os usuários
rec_percentages = pd.concat([similar_user_recs_count, all_user_recs_count], axis=1)
rec_percentages.columns = ["similar", "all"]

# Calcula o score de similaridade
rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

# Classifica os resultados por score de similaridade
rec_percentages = rec_percentages.sort_values("score", ascending=False)

# Combina os resultados com as informações dos filmes
print(rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId"))

def find_similar_movies(movie_id):
    """
    Encontra filmes similares com base nas classificações dos usuários.

    Esta função recebe um identificador de filme como entrada e encontra filmes
    que são considerados similares com base nas classificações dos usuários. A
    similaridade é determinada por meio da análise das classificações dos usuários
    que deram notas altas (maior que 4) ao filme fornecido. A função calcula a
    porcentagem de usuários que classificaram outros filmes como similares e retorna
    os filmes mais similares em ordem decrescente de similaridade.

    Args:
        movie_id (int): O identificador do filme para o qual se deseja encontrar
        filmes similares.

    Returns:
        DataFrame: Um DataFrame contendo informações sobre os filmes similares,
        incluindo pontuação de similaridade, título e gêneros.

    """
    similar_users_inner = ratings[(ratings["movieId"] == movie_id) &
                             (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs_inner = ratings[(ratings["userId"].isin(similar_users_inner)) &
                                 (ratings["rating"] > 4)]["movieId"]
    similar_user_recs_inner = similar_user_recs_inner.value_counts() / len(similar_users_inner)

    similar_user_recs_inner = similar_user_recs_inner[similar_user_recs > .10]
    all_users_inner = ratings[(ratings["movieId"].isin(similar_user_recs_inner.index)) &
                         (ratings["rating"] > 4)]
    all_user_recs_inner = all_users_inner["movieId"].value_counts() / len(
        all_users_inner["userId"].unique())
    rec_percentages_inner = pd.concat([similar_user_recs_inner, all_user_recs_inner], axis=1)
    rec_percentages_inner.columns = ["similar", "all"]

    rec_percentages_inner["score"] = rec_percentages_inner["similar"] / rec_percentages_inner["all"]
    rec_percentages_inner = rec_percentages_inner.sort_values("score", ascending=False)
    return rec_percentages_inner.head(10).merge(
        movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    """
    Limpa a saída atual e exibe recomendações de filmes semelhantes com base em um título.

    Args:
        title (str): O título do filme digitado pelo usuário.

    Returns:
        None: A função não retorna um valor diretamente, mas limpa a saída atual
        e exibe a lista de recomendações de filmes semelhantes no ambiente interativo.
    """
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            if not results.empty:
                movie_id = results.iloc[0]["movieId"]
                display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
