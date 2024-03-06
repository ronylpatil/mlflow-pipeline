
import pathlib
import yaml
import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.sklearn import log_model, load_model
from mlflow import MlflowClient
from datetime import datetime
from urllib.parse import urlparse
from dvclive import Live
from src.logger import infologger
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.base import BaseEstimator
from typing import Tuple
from pprint import pprint

infologger.info('*** Executing: train_model.py ***')
# writing import after infologger to log the info precisely 
from src.data.make_dataset import load_data

def train_model(training_feat: pd.DataFrame, y_true: pd.Series, n_estimators: int, criterion: str, max_depth: int, random_state: int, yaml_file_obj) -> Tuple[BaseEstimator, np.ndarray] :
     try : 
          model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth,
                                        random_state = random_state)
          model.fit(training_feat, y_true)
     except Exception as e :
          infologger.info(f'there\'s an issue while training model [check train_model()]. exc: {e}')
     else :
          infologger.info(f'trained {type(model).__name__} model')
          y_pred = model.predict(training_feat)
          y_pred_prob = model.predict_proba(training_feat)
          accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
          precision = metrics.precision_score(y_true, y_pred, zero_division = 1, average = 'macro')
          recall = metrics.recall_score(y_true, y_pred, average = 'macro')
          roc_score = metrics.roc_auc_score(y_true, y_pred_prob, average = 'macro', multi_class = 'ovr')

          try : 

               mlflow_config = yaml_file_obj['mlflow_config']
               # client = MlflowClient()
               # client.get_model_version_by_alias(mlflow_config['registered_model_name'], 'champion')
               
               # runs = client.search_runs(filter_string=f"tag.{'validation_status'}='{'approved'}'", run_view_type = 2)
               # print(runs)

               mlflow_config = yaml_file_obj['mlflow_config']
               remote_server_uri = mlflow_config['remote_server_uri']
               mlflow.set_tracking_uri(remote_server_uri)
               mlflow.set_experiment(mlflow_config['trainingExpName'])
               MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
               # model registery ko tag or alias dena
               client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)
               
               


     # ------------------------------------ MODEL REGISTRY ------------------------------------------------------------
               # ye particular version k registry ko ek alias assign krega
               # client.set_registered_model_alias(mlflow_config['registered_model_name'], 'looser', 20)

               # agr same alias diya kisi dusre ko to vo pichle wale se nikl ke naye wale ko assign ho jayega 
               # same alias nahi de skte, or overwrite ho jayega algr diya to
               # ab hum yha staging, production, archieve alias de skte he   
               # client.set_registered_model_alias(mlflow_config['registered_model_name'], 'looser', 9)

               # ye alias se model ki details de rha he
               # print(client.get_model_version_by_alias(mlflow_config['registered_model_name'], "looser").version)

               # alias delete kar or alias hmesha unique hoga
               # client.delete_registered_model_alias(mlflow_config['registered_model_name'], 'looser')

               # ye regertered model k version ko tag de rha he
               # client.set_model_version_tag(mlflow_config['registered_model_name'], "19", "validation_status", "archieve")

               # Delete model version tag
               # client.delete_model_version_tag(mlflow_config['registered_model_name'], "23", "validation_status")


               # ye regestered model ko tag assign karega or upar wale exm regestered model k versions ko
               # client.set_registered_model_tag(mlflow_config['registered_model_name'], "developer", "ronil")

               # Delete registered model tag
               # client.delete_registered_model_tag(mlflow_config['registered_model_name'], "task")

               # ye model version k description add/update karega
               # client.update_model_version(
               #      name = mlflow_config['registered_model_name'],
               #      version = 22,
               #      description = "this version implemented by ronil",
               # )

               # ye registered models ki partial details dega (isme bugs he abhi)
               # for i in client.search_registered_models() :
               #      for j in i :
               #           pprint(j) 
               #      print()

               # ye aliases nahi bata rah he(bug he), but ye result de rha he pura
               # print(client.search_model_versions(filter_string = "name = 'prod_testing' and tag.validation_status = 'pending'"))
# ---------------------------------------------------------------------------------------------------------------
               # search Mlflow models
               # ye registered models dega but limited info dega, latest info and ...
               # for rm in client.search_registered_models():
               #      pprint(dict(rm), indent = 4)

               # ye sare models ki info degi, yha sari details hogi
               # for mv in client.search_model_versions(f"name='{mlflow_config['registered_model_name']}'") :
               #      pprint(dict(mv), indent = 4)

               # delete registered model/version
               # versions = [1, 2, 3]
               # for version in versions :
               #      client.delete_model_version(name = mlflow_config['registered_model_name'], version = version)

               # Delete a registered model along with all its versions
               # client.delete_registered_model(name = mlflow_config['registered_model_name'])

               # as model reg stages will be deprecated so it won't make sence 
               # print(client.get_latest_versions(mlflow_config['registered_model_name']))

               # ab pura game alias karega
               # print(client.get_model_version_by_alias(mlflow_config['registered_model_name'], 'champion').run_id)

# --------------------------------------------------------------------------------------------------------------------------------
               # ye bhi kaam kr rha he
               from mlflow.pyfunc import load_model

               # different reg models k versions ko ek ek alias tag kr do fir 
               #  us alias se uska version pta karo or version se model load karo thats it.
               model_version = client.get_model_version_by_alias(mlflow_config['registered_model_name'], 'champion').version
               # model = load_model(f"models:/{mlflow_config['registered_model_name']}/{model_version}")
               model = load_model(f"models:/{mlflow_config['registered_model_name']}/{model_version}")
               
               inp = [[8.207097522651933,0.3651906838774485,0.378633478714214,4.542055303465597,
               0.079792372644902,25.448093161225515,38.8633478714214,0.9945203178994676,
               3.23103813677549,0.6893432309794074,12.555190683877449,8.950921685243596,
               2.7702927995076014,0.6548095970892682,1.4026701523459664,11.995915730668555,
               12.624368208379435,15.786228820652939,5.311190907089906]]
               print(model.predict(inp))
               # NOTe: predict_proba is not available in pyfunc.load_model instead go for mlflow.sklearn.load_model
               print(model.predict_proba(inp))



          except Exception as ie : 
               infologger.info(f'there\'s an issue while tracking metrics/parameters using mlflow [check train_model()]. exc: {ie}')
          else : 
               infologger.info('parameters/metrics tracked by mlflow')
               return model, y_pred

def save_model(model: BaseEstimator, model_dir: str) -> None : 
     try : 
          joblib.dump(model, f'{model_dir}/model.joblib')
     except Exception as e : 
          infologger.info(f'there\'s an issue while saving the model [check save_model(). exc: {e}')
     else :
          infologger.info(f'model saved at {model_dir}')

def main() -> None : 
     curr_path = pathlib.Path(__file__) 
     home_dir = curr_path.parent.parent.parent
     params_loc = f'{home_dir.as_posix()}/params.yaml'
     try : 
          params = yaml.safe_load(open(params_loc, encoding = 'utf8'))
     except Exception as e :
          infologger.info(f'there\'s an issue while loading params.yaml [check main()]. exc: {e}')
     else : 
          parameters = params['train_model']
          TARGET = params['base']['target']

          train_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_train.csv"
          model_dir = f"{home_dir.as_posix()}{parameters['model_dir']}"
          pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
          
          data = load_data(train_data)
          X_train = data.drop(columns = [TARGET]).values
          Y = data[TARGET]

          model, _ = train_model(X_train, Y, parameters['n_estimators'], parameters['criterion'], parameters['max_depth'], parameters['seed'], yaml_file_obj = params)
          # save_model(model, model_dir)
          infologger.info('program terminated normally!')

if __name__ == '__main__' : 
     infologger.info('train_model.py as __main__')
     main()
     