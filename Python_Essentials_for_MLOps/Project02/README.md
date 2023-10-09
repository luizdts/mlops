criar virtual env airflow
pip install virtualenv
python3.10 -m venv airflow

source airflow/bin/activate
$ CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-2.7.1/constraints-3.10.txt"
$ pip install "apache-airflow==2.7.1" --constraint "${CONSTRAINT_URL}"

pip install apache-airflow=='2.7.1'


airflow standalone

deactivate