import yaml
import pathlib
import pandas as pd
from src.logger import infologger
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

infologger.info("*** Executing: make_dataset.py ***")

# load data
def load_data(file_path: str) -> pd.DataFrame : 
    try : 
        df = pd.read_csv(file_path)
    except Exception as e : 
        infologger.info(f'there\'s some issue while loading data [check load_data()]. exc: {e}')
    else : 
        infologger.info(f'data loaded from {file_path}')  
        return df

# perform basic preprocessing
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame : 
    try : 
        data = data.drop('Id', axis = 1)
    except Exception as e : 
        infologger.info(f'there\'s some issue while preprocessig the data [check preprocess_data()]. exc: {e}')
    else : 
        infologger.info('data preprocesed successfully')
        return data

def oversampling(X: pd.DataFrame, y: pd.Series, target: str, seed: int) -> pd.DataFrame : 
    try : 
        smote = SMOTE(random_state = seed)
        X_res, y_res = smote.fit_resample(X, y)
    except Exception as e : 
        infologger.info(f'there\'s an issue while performing over-sampling [check oversampling()]. exc: {e}')
    else : 
        X_res[target] = y_res
        infologger.info('data oversampled using SMOTE!')
        return X_res

# split the data
def split_data(test_split: int, seed: int, data: pd.DataFrame) -> pd.DataFrame : 
    try : 
        train, test = train_test_split(data, test_size = test_split, random_state = seed)
    except Exception as e : 
        infologger.info(f'there\'s some issue while spliting the data [check split_data()]. exc: {e}')
    else : 
        infologger.info(f'data splited with test_split: {test_split} & seed: {seed}')
        return train, test

# save the data
def save_data(path: str, train: pd.DataFrame, test: pd.DataFrame) -> None : 
    try : 
        train.to_csv(f'{path}/train.csv', index = False)
        test.to_csv(f'{path}/test.csv', index = False)
    except Exception as e : 
        infologger.info(f'there\'s some issue while saving the data [check save_data()]. exc: {e}')
    else :
        infologger.info(f'training data saved at {path}')
        infologger.info(f'testing data saved at {path}')

def main() -> None : 
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    params_file_loc = f'{home_dir.as_posix()}/params.yaml'
    try : 
        params = yaml.safe_load(open(params_file_loc, encoding = 'utf8'))
    except Exception as e : 
        infologger.info(f'there\'s some issue while loading params.yaml [check main()]. exc: {e}')
    else :
        parameters = params['make_dataset']
        TARGET = params['base']['target']

        # loading the data from data/raw dir
        file_path = f'{home_dir.as_posix()}{params["load_dataset"]["raw_data"]}/{params["load_dataset"]["file_name"]}.csv'
        data = preprocess_data(load_data(file_path))
        # oversampling() ko data de or ye new aage pass kr
        
        X = data.drop(columns = [TARGET])
        y = data[TARGET]
        data_res = oversampling(X = X, y = y, target = TARGET, seed = parameters['res_seed'])
        train, test = split_data(parameters['test_split'], parameters['seed'], data = data_res)
        path = f'{home_dir.as_posix()}{parameters["processed_data"]}'
        save_data(path = path, train = train, test = test)
        infologger.info('program terminated normally!')

if __name__ == "__main__" :
    infologger.info('make_dataset.py as __main__')
    main()
