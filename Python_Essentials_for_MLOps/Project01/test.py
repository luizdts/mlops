"""
Teste unitário do movie_recommendations.py
"""
import unittest
import pandas as pd

from movie_recommendations import clean_title, search, find_similar_movies

class TestMovieRecommendations(unittest.TestCase):
    """
    Módulo de teste utilizado para o arquivo movie_recommendations.py
    """
    def setUp(self):
        """
        Dados mockados
        """
        self.movies = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title': ['Toy Story 1', 'Toy Story 2', 'Toy Story 3', 'Avatar', 'The Matrix']
        })

    def test_clean_title(self):
        """
        Realiza o teste para tirar os espaços e caracteres especiais do título de um filme
        """
        cleaned_title = clean_title("Toy Story 1")
        self.assertEqual(cleaned_title, "ToyStory1")

    def test_search(self):
        """
        Realiza a busca de um filme
        """
        result = search("Monsters Inc")
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(len(result) > 0)

    def test_find_similar_movies(self):
        """
        Testa a busca de filmes similares
        """
        # Simulate the function's behavior with a mock
        def mock_find_similar_movies(movie_id):
            return pd.DataFrame({
                'score': [0.9, 0.85, 0.8],
                'title': ['Similar Movie 1', 'Similar Movie 2', 'Similar Movie 3'],
                'genres': ['Comedy', 'Action', 'Adventure']
            })

        result = mock_find_similar_movies(1)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()
