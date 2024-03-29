import pytest
import pathlib
import yaml
from fastapi.testclient import TestClient
from prod.api import app
from src.data.load_dataset import extract_data
from src.features.build_features import feat_eng
from src.data.make_dataset import load_data
from src.models.train_model import train_model

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent
params = yaml.safe_load(open(f"{home_dir.as_posix()}/params.yaml"))
drive_ = params['load_dataset']['drive_link']
processed_data = params['make_dataset']['processed_data']
data_source = f'{home_dir.as_posix()}{processed_data}/train.csv'
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

# fixture with params
# return the output of extract_data function
@pytest.fixture(params = [drive_])
def load_dataset(request) :
     # calling extract_data method & returning its output 
     return extract_data(request.param)

# return output of feat_eng function
@pytest.fixture(params = [data_source])
def load(request) : 
     return feat_eng(load_data(request.param))

# fixture without parameters
# return output of train_model function
@pytest.fixture
def get_model() :
     return train_model(X_train, Y, parameters['n_estimators'], parameters['criterion'], parameters['max_depth'], min_samples_leaf = parameters['min_samples_leaf'],
                 min_samples_split = parameters['min_samples_split'], random_state = parameters['random_state'], yaml_file_obj = params)

# TestClient(app) is used to create a test client object for simulating/
#  HTTP requests to a FastAPI application during testing
@pytest.fixture
def client() :
     return TestClient(app)
