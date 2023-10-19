# Python Essentials for MLOps

## Introdução

Esse repositório contém todos os projetos desenvolvidos na primeira unidade da disciplina de Projeto de Sistemas Baseados em Aprendizado de Máquina - DCA0305, nessa unidade tivemos como foco principal aplicar ferramentas e conceitos de código limpo (DRY e KISS), criar logging para as aplicações que foram desenvolvidas, testes unitários e descrever sucintamente como cada projeto é executado e seus resultados finais.

Para essa etapa, foram realizados três projetos, um que envolve um sistema de recomendação de filmes, um projeto que realiza o download de episódios de podcasts gerenciado em conjunto ao Airflow e o último projeto que é voltado para otimização de modelos de Machine Learning em Python.


## Descrição dos projetos realizados

- Projeto 01: Build a Movie Recommendation System in Python
    - Esse projeto têm como objetivo criar um sistema de recomendação de filmes utilizando Python, é dado dois datasets "movies.csv" e "ratings.csv" que possuem métricas com o nome dos filmes e as suas respectivas avaliações. 
- Projeto 02: Build an Airflow Data Pipeline to Download Podcasts
    - Esse projeto têm como objetivo principal utilizar a ferramenta de DAGs Airflow para que seja feito uma série de tarefas para que seja possível obter episódios de podcast baixados e transcritos, o Airflow é responsável por criar o ambiente onde será executada essa DAG e um arquivo Python no qual possui toda a configuração das task's que irão ser utilizadas pelo Airflow. 
- Projeto 03: Optimizing Machine Learning Models in Python
    - Esse projeto têm como objetivo realizar a otimização de um modelo de Machine Learning, utilizando Sklearn como a biblioteca para aplicar funções específicas (Regressão Linear, LassoCV, RidgeCV, KNNImputer) no código, o arquivo Python têm como objetivo prever a extensão dos danos de uma queimada e assim prever de acordo com características os possíveis danos possíveis de uma futura queimada, tudo isso a partir do conjunto de dados "fires.csv".

Atividades realizadas nos projetos:
- Refatoração de Código
- Princípios de Código Limpo
- Linting
- Lançamento de Exceptions
- Testes Unitários

Repositório dos projetos realizados:
- Project 01 - [Build a Movie Recommendation System in Python](https://github.com/luizdts/mlops2023/tree/main/Python_Essentials_for_MLOps/Project01)
- Project 02 - [Build an Airflow Data Pipeline to Download Podcasts](https://github.com/luizdts/mlops2023/tree/main/Python_Essentials_for_MLOps/Project02)
- Project 03 - [Optimizing Machine Learning Models in Python](https://github.com/luizdts/mlops2023/tree/main/Python_Essentials_for_MLOps/Project03)

Certificado do curso Intermediate Python for Web Development [Link para o certificado](https://app.dataquest.io/view_cert/NIZH7IQPJ2XN1MXZ7227) - [Verificador](https://app.dataquest.io/verify_cert/NIZH7IQPJ2XN1MXZ7227/)


Video sobre os projetos disponível na plataforma [Loom](https://www.loom.com/share/cf65bbc12bd14b82941689ddc54e915a?sid=50359c82-39df-4980-8379-725ec720f7a5)

## Tecnologias utilizadas 

Para desenvolver e aprimorar as aplicações, foram utilizadas as seguintes tecnologias:
- Github Codespaces
- Python 3.8+
- Airflow 2.3+
- Logging
- Pylint
- Pytest
- Sklearn
- SQLite3

## Instalação

Para executar o ambiente, devemos nos atentar à instalação dos pacotes necessários para cada projeto, com isso, recomenda-se o uso do pip para que todos os pacotes sejam corretamente instalados.

```bash
pip install nome_do_pacote
```

Para clonar o repositório e ter acesso aos arquivos, pode ser utilizado a seguinte linha de comando:
```bash
git clone https://github.com/luizdts/mlops2023.git
```

## Referências: 
- Transparências do professor Ivanovitch Medeiros
- Dataquest.io
- Stack Overflow
- ChatGPT