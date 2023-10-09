# -*- coding: utf-8 -*-
"""
    UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE
    DEPARTAMENTO DE COMPUTAÇAO E AUTOMAÇAO

    DISCENTE: LUIZ HENRIQUE ARAUJO DANTAS
    PROJETO 3 - OPTIMIZING MACHINE LEARNING MODELS IN PYTHON

"""

import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV, RidgeCV

logging.basicConfig(filename='fires.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    fires = pd.read_csv("./fires.csv")
    logging.info("O arquivo CSV 'fires.csv' foi lido com sucesso.")
except FileNotFoundError as e:
    logging.error("Erro: %s. O arquivo CSV 'fires.csv' não foi encontrado.", e)
except pd.errors.EmptyDataError as e:
    logging.error("Erro: %s. O arquivo CSV 'fires.csv' está vazio.", e)
except pd.errors.ParserError as e:
    logging.error("Erro: %s. Não foi possível analisar o arquivo CSV 'fires.csv'.", e)

columns_of_interest = ["wind", "temp", "area"]
fires_reference = fires[columns_of_interest].dropna()

reference_X = fires_reference[["wind", "temp"]]
reference_y = fires_reference["area"]

reference_model = LinearRegression()
reference = LinearRegression()

def count_missing_values(df, col_name):
    """
    Conta o número de valores ausentes em uma coluna específica de um DataFrame.

    Esta função recebe um DataFrame e o nome de uma coluna (col_name) e calcula a
    quantidade de valores ausentes (NaN) nessa coluna.
    """
    return sum(pd.isna(df[col_name]))

for col in fires.columns:
    num_na = count_missing_values(fires, col)
    print(f"The {col} column has {num_na} missing values.")

# Plotar o histograma da coluna "area"
fires.hist("area", bins=30)

# Calcular o log da coluna "area" e adicionar ao DataFrame
fires["log_area"] = np.log(fires["area"] + 1)

# Plotar o histograma do log da coluna "area"
fires.hist("log_area", bins=30)

def is_summer_month(month):
    """
    Verifica se um mês é considerado um mês de verão.

    Esta função recebe o nome abreviado de um mês e verifica se ele está na lista
    de meses de verão, que inclui junho (jun), julho (jul) e agosto (aug).

    Parâmetros:
    month (str): O nome abreviado do mês a ser verificado.

    Retorna:
    int: Retorna 1 se o mês for um mês de verão (jun, jul, ou aug), caso contrário, retorna 0.

    Exemplo:
    >>> is_summer_month("jun")
    1
    >>> is_summer_month("set")
    0
    """
    month = month.lower()
    if month in ["jun", "jul", "aug"]:
        return 1
    return 0

fires["summer"] = [is_summer_month(m) for m in fires["month"]]

imp = KNNImputer(missing_values = np.nan, n_neighbors=3)

fires_missing = fires[fires.columns[5:13]]

imputed = pd.DataFrame(imp.fit_transform(fires_missing),
                       columns = fires.columns[5:13])

imputed.boxplot(column=["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"])

for col in imputed:

    quartiles = np.percentile(fires[col], [25, 50, 75])
    iqr = quartiles[2] - quartiles[0]
    lower_bound = quartiles[0] - (1.5 * iqr)
    upper_bound = quartiles[2] + (1.5 * iqr)
    num_outliers =sum((imputed[col] < lower_bound) | (imputed[col] > upper_bound))

    print(f"The {col} column has {num_outliers} according to the boxplot method.")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed)
scaled_df = pd.DataFrame(scaled_data, columns=fires.columns[5:13])

final = pd.concat([fires["summer"], scaled_df], axis=1)

y = fires["log_area"]

# Função para criar modelos de regressão linear com seleção de características
def create_linear_regression_model(n_features_to_select):
    """
    Cria um modelo de regressão linear com seleção de características.

    Esta função cria um modelo de regressão linear com seleção de características
    usando SequentialFeatureSelector da biblioteca scikit-learn. O modelo é treinado
    para selecionar um número específico de características, conforme especificado
    pelo argumento 'n_features_to_select'.
    """
    sfs_model = LinearRegression()
    sfs = SequentialFeatureSelector(estimator=sfs_model,
                                    n_features_to_select=n_features_to_select,
                                    direction="forward")
    sfs.fit(final, y)
    selected_features = sfs.get_feature_names_out()
    regression_model = LinearRegression()  # Crie um modelo de regressão linear
    return  regression_model, selected_features

# Crie modelos com diferentes números de características selecionadas
fw2_model, fw2_features = create_linear_regression_model(2)
logging.info("Forward-2 Model features: %s", fw2_features)

fw4_model, fw4_features = create_linear_regression_model(4)
logging.info("Forward-4 Model features: %s", fw4_features)

fw6_model, fw6_features = create_linear_regression_model(6)
logging.info("Forward-6 Model features: %s", fw6_features)

bw2_model, bw2_features = create_linear_regression_model(2)
logging.info("Backward-2 Model features: %s", bw2_features)

bw4_model, bw4_features = create_linear_regression_model(4)
logging.info("Backward-4 Model features: %s", bw4_features)

bw6_model, bw6_features = create_linear_regression_model(6)
logging.info("Backward-6 Model features: %s", bw6_features)

print("Features selected in 2 feature model:", fw2_features)
print("Features selected in 4 feature model:", fw4_features)
print("Features selected in 6 feature model:", fw6_features)
print("Features selected in 2 feature model:", bw2_features)
print("Features selected in 4 feature model:", bw4_features)
print("Features selected in 6 feature model:", bw6_features)

ridge = RidgeCV(alphas = np.linspace(1, 10000, num=1000))
lasso = LassoCV(alphas = np.linspace(1, 10000, num=1000))

ridge.fit(final, y)
lasso.fit(final, y)

print("Ridge tuning parameter: ", ridge.alpha_)
print("LASSO tuning parameter: ", lasso.alpha_)

print("Ridge coefficients: ", ridge.coef_)
print("LASSO coefficients: ", lasso.coef_)

# Crie um único objeto RidgeCV para ajustar o hiperparâmetro alpha
ridge_cv = RidgeCV(alphas=np.linspace(1000, 1500, num=1000))
ridge_cv.fit(final, y)
print("Ridge tuning parameter:", ridge_cv.alpha_)

# Lista de modelos a serem avaliados
models = [reference, fw2_model, fw4_model, fw6_model,
           bw2_model, bw4_model, bw6_model, ridge_cv]
model_names = ["Reference", "Forward-2", "Forward-4",
               "Forward-6", "Backward-2", "Backward-4", "Backward-6", "Ridge"]

# Avalie cada modelo com validação cruzada
for model, name in zip(models, model_names):
    cv_scores = cross_val_score(model, final[["wind", "temp"]],
                                 y, cv=5, scoring="neg_mean_squared_error")
    avg_mse = -np.mean(cv_scores)
    std_mse = np.std(cv_scores)
    print(f"{name} Model, Avg Test MSE: {avg_mse:.2f}, SD: {std_mse:.2f}")
