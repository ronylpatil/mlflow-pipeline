# data is properly loading from source or not
# check build_features working or not    ---- done
# check model training is working fine or not and model return kr rha he ki nahi
# model tunning performance improve kr rha he ki nahi
# check whether visualization is working fine or not, or plot return kr rha he ki nahi
# check api working fine or not, response kya de rha he api glt or sahi input pr
# check streamlit working fine or not,

import pathlib
import pytest
import yaml
from src.features.build_features import feat_eng
from src.data.make_dataset import load_data

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent

params = yaml.safe_load(open(f"{home_dir.as_posix()}/params.yaml"))

processed_data = params['make_dataset']['processed_data']
data_source = f'{home_dir.as_posix()}{processed_data}/train.csv'


# 'request' is a fixture that provides information about the currently executing test.
# It allows us to access various attributes and methods to get information about the test
# environment, access fixtures, and more.
# here we are accessing current parameters value using request.param
@pytest.fixture(params = [data_source])
def load(request) : 
     return feat_eng(load_data(request.param))

def test_build_features(load) : 
     assert load.shape[1] == 20



'''
tox.ini file initially khud se create karo or configure karo
but ek cheez ka dhyan de ki agr test_files kisi dir me hoto kese tox.ini me mention krna he
or khas baat requirements.txt dhyan se dena, local package bhi mention krna
bss fix requirements.txt reload kr k testing kran hoto
cmd: tox -r
else cmd: tox
'''
