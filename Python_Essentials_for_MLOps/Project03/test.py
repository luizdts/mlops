"""
    Unit Tests
"""
import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from optimizingmodel import *

class TestFireAnalysis(unittest.TestCase):
    """
        Class used to create the tests
    """
    def setUp(self):
        """
        Mocked values
        """
        self.fires = pd.read_csv("fires.csv")
        self.columns_of_interest = ["wind", "temp", "area"]

    def test_read_csv(self):
        """
        Tests read csv
        """
        self.assertIsNotNone(self.fires)
        self.assertEqual(len(self.fires), 517)

    def test_is_summer_month(self):
        """
        Test if is a summer month
        """
        self.assertEqual(is_summer_month("jun"), 1)
        self.assertEqual(is_summer_month("jul"), 1)
        self.assertEqual(is_summer_month("aug"), 1)
        self.assertEqual(is_summer_month("set"), 0)
        self.assertEqual(is_summer_month("JUN"), 1)  # Teste com maiúsculas
        self.assertEqual(is_summer_month("invalid_month"), 0)  # Mês inválido

    def test_create_linear_regression_model(self):
        """
        Test if is created a linear regression model
        """
        model_test, selected_features = create_linear_regression_model(2)
        self.assertIsInstance(model_test, LinearRegression)
        self.assertEqual(len(selected_features), 2)
if __name__ == "__main__":
    unittest.main()
