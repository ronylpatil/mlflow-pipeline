import pathlib
import pytest
import yaml
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator
from prod.api import app
from src.features.build_features import feat_eng
from src.data.make_dataset import load_data
from src.data.load_dataset import extract_data
from src.models.train_model import train_model
from src.visualization.visualize import conf_matrix


curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent

params = yaml.safe_load(open(f"{home_dir.as_posix()}/params.yaml"))

processed_data = params['make_dataset']['processed_data']
data_source = f'{home_dir.as_posix()}{processed_data}/train.csv'

drive_ = params['load_dataset']['drive_link']
parameters = params['train_model']
TARGET = params['base']['target']
train_dir = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_train.csv"
test_dir = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"
model_dir = f"{home_dir.as_posix()}{parameters['model_dir']}"
pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
train_data = load_data(train_dir)
test_data = load_data(test_dir)
X_train = train_data.drop(columns = [TARGET]).values
Y = train_data[TARGET]
X_test = test_data.drop(columns = [TARGET]).values
y = test_data[TARGET]

# test api endpoint
# valid input
user_input1 = {  
                'fixed_acidity': 7.6,
                'volatile_acidity': 0.715,
                'citric_acid': 0,
                'residual_sugar': 15.1,
                'chlorides': 0.068,
                'free_sulfur_dioxide': 30,
                'total_sulfur_dioxide': 35,
                'density': 0.99533,
                'pH': 3.48,
                'sulphates': 0.65,
                'alcohol': '11.4'}

# invalid input - remove few columns
user_input2 = {  
                'fixed_acidity': 9.9384,
                'volatile_acidity': 0.6412,
                'citric_acid': 0.4372,
                'residual_sugar': 5.4216,
                'chlorides': 0.0719,
                'free_sulfur_dioxide': 7.2126,
                'total_sulfur_dioxide': 12.7873,
                'density': 0.997,
                'pH': 3.255,
                'sulphates': 0.7923 }

# invalid input - values out-of-range
user_input3 = {  
                'fixed_acidity': 700.6,
                'volatile_acidity': 0.715,
                'citric_acid': 780,
                'residual_sugar': 15.1,
                'chlorides': 0.068,
                'free_sulfur_dioxide': 3000,
                'total_sulfur_dioxide': 335,
                'density': 10.99533,
                'pH': 33.48,
                'sulphates': 0.65,
                'alcohol': '11.4'}

# test load_data function
def test_load_data(load_dataset) :
     # load_dataset is pointing to the dataset returned by extract_data method 
     if isinstance(load_dataset, pd.DataFrame) :            
          assert isinstance(load_dataset, pd.DataFrame)      # checking load_dataset is instance of pandas dataframe, if yes testcase will pass
     else : 
          pytest.fail('Unable to fetch data from remote server')     # else test case will fail and return the message

     # ---- OR ----
     # assert isinstance(load_dataset, pd.DataFrame), "Failure Message"     # it will raise assertion error 

# test build_features function
def test_build_features(load) : 
     if load.shape[1] == 20 :
          assert load.shape[1] == 20
     else :
          pytest.fail('build_feature is not working properly')

# test train_model function
def test_train_model(get_model) :
     if isinstance(get_model['model'], BaseEstimator) :
          assert isinstance(get_model['model'], BaseEstimator)
     else :
          pytest.fail('not able to train the model')

# test conf_metrix function
def test_conf_metrix(get_model) :
     filename = conf_matrix(y_test = y, y_pred = get_model['model'].predict(X_test), labels = get_model['model'].classes_, path = './tests/trash', params_obj = params)
     if pathlib.Path(filename).exists() :
          assert pathlib.Path(filename).exists()
     else :
          pytest.fail('unable to generate confusion matrix')

# test API response
def test_response(client) :
     response = client.get("/")
     if response.status_code == 200 : 
          assert response.status_code == 200
          assert response.json() == {'api status': 'up & running'}
     else : 
          pytest.fail('API is not responding')

# test api endpoint with valid input
def test_validIp(client) :
     response = client.post("/predict", json = user_input1)
     if response.status_code == 200 : 
          assert response.status_code == 200
          assert 'predictions' in response.json()
          assert 'probability' in response.json()
     else : 
          pytest.fail('"predict/" endpoint failed on valid input')

# test api endpoint with invalid inputs
def test_invalidIp(client) :
     response1 = client.post("/predict", json = user_input2)
     response2 = client.post("/predict", json = user_input3)
     if response1.status_code == 422 and response2.status_code == 422 :  
          assert response1.status_code == 422
          assert response2.status_code == 422
     else :
          pytest.fail('"predict/" endpoint failed on invalid input')

# test api endpoint output
def test_details(client) :
     response = client.get("/details")
     if response.status_code == 200 : 
          assert 'model name' in response.json()
          assert 'model run_id' in response.json()
          assert 'model version' in response.json()
          assert 'model aliases' in response.json()
     else :
          pytest.fails('"details/" endpoint not responding')

'''
create & configure tox.ini file
Commands : 
     cmd: tox 
     cmd: tox -r (reload the packages)
'''

# pytest cmd to test single function
# cmd: "pytest tests/test_main.py::test_build_features --verbose --disable-warnings"
