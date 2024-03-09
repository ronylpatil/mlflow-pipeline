
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

# ------------------------------------- model ko experiment se prod env me serve krne tk ki kahani--------------------

# Plan of Action
# as of now 
# 1. hum experiment ka description add ke paye he code set
# 2. each runs ka description add kr paye help
# 3. har run ko ek tag de paye he

# Next target 
# pick experiment and from experiments pick run based of condition -- done
# register model (new/existing) -- done
# model registry ka description  -- done
# model reg ko tag  -- done
# version ko ek description -- done 
# version ko ek tag   -- done
# verion ko ek alias  -- done

# get all versions details of registered model

import pathlib
import mlflow
import yaml
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


curr_path = pathlib.Path(__file__) 
home_dir = curr_path.parent.parent.parent
params_loc = f'{home_dir.as_posix()}/params.yaml'
params = yaml.safe_load(open(params_loc, encoding = 'utf8'))

mlflow.set_tracking_uri(params['mlflow_config']['mlflow_tracking_uri'])
client = MlflowClient() 

# get version no if alias is present else get None

# print(client.get_model_version_by_alias(name = 'api_testing', alias = 'production').version)

# try : 
#      version = client.get_model_version_by_alias(name = 'api_testing', alias = 'productions').version
# except MlflowException :
#      version = None






'''
problem statement :
     experiments k run mese 1 efficient model nikalo -- done
     model register karo -- done
     description or tag assign karo  -- done
     version ko description, tag and alias assign karo (alias production rakho)  -- done

     agr prod me koi nahi he to isko dalo else agr he already to
     jo model production me already he uski accuracy current se km hoto
     usko archieve me dalo or current ko production me dalo 
     else message dalo ki efficient model already production me he

'''

exp_name = 'modeltunning'

# naam se id nikalo
exp_id = client.search_experiments(filter_string = f"name = '{exp_name}'")[0].experiment_id
print('Experiment id mil gyi:', exp_id)


# ab id se uske runs search karo
# for i in client.search_runs(experiment_ids = exp_id, order_by = ['metrics.accuracy DESC', 'metrics.precision DESC', 'metrics.recall DESC']) : 
#      # print(i, '\n')
#      print(f"accuracy: {i.data.metrics['accuracy']:.4f} | precision: {i.data.metrics['precision']:.4f} | recall: {i.data.metrics['recall']:.4f} | \
# run_name: {i.info.run_name} | run_id: {i.info.run_id}")
     

# print(client.search_runs(experiment_ids = exp_id, filter_string = 'metrics.accuracy > 0.74', order_by = ['metrics.precision DESC', 'metrics.recall DESC'])[0])

# top-1 ki run id nikalo
new_model = client.search_runs(experiment_ids = exp_id, order_by = ['metrics.accuracy DESC', 'metrics.precision DESC', 'metrics.recall DESC'])[6]
run_id = new_model.info.run_id
reg_model_name = 'model1'

if client.search_registered_models(filter_string = f"name = '{reg_model_name}'") :     # model already register he
     print('model already register he')
     try : 
          prod_version = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').version
          print('prod me already ek model serve kr rha he')
     except MlflowException as e : 
          # prod_version = None
          print('Exception: ', e)
          print('Exception aaya re baba')
     else :
          prod_run_id = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').run_id
          prod_accuracy = mlflow.get_run(run_id = prod_run_id).data.metrics['accuracy']
          print('prod model ki run id or acc mil gyi')

          if new_model.data.metrics['accuracy'] > prod_accuracy :
               print('new model outperform kr rha he, lets put into prod env') 
               # model mil gya ab model register karo
               model_uri = f"runs:/{run_id}/sklearn-model"
               mv = mlflow.register_model(model_uri, name = reg_model_name, tags = {'dev': 'ronil', 'val_status': 'approved'})
               print('model register ko register kr diya')

               # register model ka desc update karo
               client.update_registered_model(name = reg_model_name, description = 'random forest model with acc > 75%')
               print('reg model ka desc bhi add kr diya')

               # register model me tag dalna pdega
               client.set_registered_model_tag(name = reg_model_name, key = 'approved_by', value = 'SMLE-II')
               print('register model ko tag bhi de diya')

               # version ko description dega
               client.update_model_version(name = reg_model_name, version = mv.version, description = 'version-I by dev ronil')
               print('ab version ka description add kr diya')

               client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production')
               client.set_registered_model_alias(name = reg_model_name, version = prod_version, alias = 'archive')
               # client.delete_model_version_tag(reg_model_name, version = prod_version, key = 'val_status')
               client.set_model_version_tag(name = reg_model_name, version = prod_version, key = 'val_status', value = 'trash')
          else : 
               print('High performing model already in production.')
else :     # model register hi nahi he 
     # model mil gya ab model register karo
     model_uri = f"runs:/{run_id}/sklearn-model"
     mv = mlflow.register_model(model_uri, name = reg_model_name, tags = {'dev': 'ronil', 'val_status': 'approved'})
     print('model register kr diya')

     # register model ka desc update karo
     client.update_registered_model(name = reg_model_name, description = 'random forest model with acc > 75%')
     print('reg model ka desc bhi add kr diya')

     # register model me tag dalna pdega
     client.set_registered_model_tag(name = reg_model_name, key = 'approved_by', value = 'SMLE-II')
     print('register model ko tag bhi de diya')

     # version ko description dega
     client.update_model_version(name = reg_model_name, version = mv.version, description = 'version-I by dev ronil')
     print('ab version ko description de diya')

     client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production')
     print('first version he to direct prod me dal diya')



# -------------------------------------------------------------------------------------------------------------------------------------------------------

'''
if prod_version : 

     prod_run_id = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').run_id

     prod_accuracy = mlflow.get_run(run_id = prod_run_id).data.metrics['accuracy']

     if x.data.metrics['accuracy'] > prod_accuracy : 
          # model mil gya ab model register karo
          model_uri = f"runs:/{run_id}/sklearn-model"
          mv = mlflow.register_model(model_uri, name = reg_model_name, tags = {'dev': 'ronil'})
          print('model register bhi ho gya')

          # register model ka desc update karo
          client.update_registered_model(name = reg_model_name, description = 'random forest model with acc > 75%')
          print('reg model ka desc bhi add kr diya')

          # register model me tag dalna pdega
          client.set_registered_model_tag(name = reg_model_name, key = 'author', value = 'ronilpatil')
          client.set_registered_model_tag(name = reg_model_name, key = 'approved_by', value = 'SMLE-II')
          print('register model ko tag bhi de diya')

          # version ko description dega
          client.update_model_version(name = reg_model_name, version = mv.version, description = 'version-I by dev ronil')
          print('ab version ko description de diya')

          if mv.version == 1 : 
               # tag already de diya register krte time, ab alias ki bari
               client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production') 
               print('first version he to direct register kr diya')
          else : 
               print('else condition me ghusa')'''

# ------------------------------------------------------------------------------------------------------------------------------------------------

'''
try : 
     prod_version = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').version
     prod_run_id = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').run_id
     print('production me model already he bro')
     print(f"version: {prod_version} and run_id: {prod_run_id}")
except MlflowException :
     prod_version = None
     print('koisa bhi model prod env me he hi nahi')

if prod_version : 
     prod_accuracy = mlflow.get_run(run_id = prod_run_id).data.metrics['accuracy']
     accuracy = mlflow.get_run(run_id = run_id).data.metrics['accuracy']
     if prod_accuracy > accuracy : 
          # delete th registered model
          client.delete_model_version(name = reg_model_name, version = mv.version)
          print(f"prod accuracy: {prod_accuracy} > new_model_acc {accuracy}")
          print('No need to update the production model')
          print('registered model delete kr diya')
     else : 
          client.set_registered_model_alias(name = reg_model_name, version = prod_version, alias = 'archive')
          client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production')
          print('Model prod me dal gya bhidu, chill kr')
else :
     client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production')'''

# -------------------------------------------------------------------------------------------------------------------------------

# ye experiments ki sari details dega
# for i in client.search_experiments() :
     # print(i)

# ye individual experiemnt ki detail dega, exp id is mandatory
# print(client.get_experiment(experiment_id = 3))

# ye particular exp k sare runs dega or condition se filtering bhi possible he
# for i in client.search_runs(experiment_ids = 2): 
     # print(i, '\n\n')

# ye particular experiment k runs ko filter kr k dega, yha condition lgayi he or sort b
# print(len(client.search_runs(experiment_ids = 2, filter_string = 'metrics.accuracy > 0.73', order_by = ['metric.accuracy DESC'])))


# --------------------------- model registry --------------------------------------------------


# ------------------------------- model level pr

# ye model register karega, pr humko run id dena hogi kisi bhi run ki, ye hum search_runs se le skte he 
# agar same name se already reg he to usi me ek version create ke dega other wise new create karega
# model_uri = f"runs:/{'ae57325ad7ed45d9bc08197c9913cda6'}/sklearn-model"
# # yha ye tags, version ko tag dega, naki registered model ko
# mv = mlflow.register_model(model_uri = model_uri, name = 'api_testing', tags = {'author':'ronil', 'quarter': 'Q1 2024'})
# # various parameters r there we can check
# print(mv.version)
# print(mv.aliases)

# agar registered model already exist krte hoto new version create krega ye
# is case me tags kaam aaya, new version create kiya isne or usko tags assign kiya
# mv = mlflow.register_model(model_uri = f"runs:/{'641f57a277ad412584f925ea0b7b8655'}/sklearn-model", name = 'api_testing', tags = {'validation status': 'in-approval'})

# ------------------------ description
# ye particular registered model ka description add karega
# client.update_registered_model(name = 'api_testing', description = 'adding this description explicitely')

# ---------------------- tags
# ye registered model ko tag assign karega, multiple tag at a time assign nahi kr skte one by one krna pdega
# client.set_registered_model_tag(name = 'api_testing', key = 'author', value = 'ronil')

# Delete registered model ka tag
# client.delete_registered_model_tag(mlflow_config['registered_model_name'], "task")

# ----------------------- get details
# ye registered models ki details dega, iska use kr k hum kuch bhi kr skte he
# for i in client.search_registered_models() :
#      for j in i :
#           print(j) 
#      print()

# ----------------- delete
# Delete a registered model along with all its versions
# client.delete_registered_model(name = mlflow_config['registered_model_name'])


# ----------------------------------------- version/run level pr 

# --------------------- search version
# isse hum kisi bhi registered model k versions ko filter kr skte he but isme ek bug bol skte he 
# ki ye us version ka alias extract nhi kr pa rha he
# print(client.search_model_versions(filter_string = "name = 'api_testing' and tags.status = 'pending'"))

# ----------------- description
# ye kisi bhi registered model k version ka description add/modify kr skta he
# client.update_model_version(
     # name = 'api_testing',
     # version = 2,
     # description = 'adding description for version-2 of api_tesing registered model',
# )

# ------------------- tags
# ye kisi bho reg model k version ko tag assign karega
# client.set_model_version_tag(name = 'api_testing', version = 1, key = 'status', value = 'pending')

# Delete model version tag
# client.delete_model_version_tag('api_testing', '23', 'validation_status')


# ------------------------ aliases

# client.set_registered_model_alias(name = 'api_testing', alias = 'staging', version = 1)

# ye version k alias se model ki detail dega
# print(client.get_model_version_by_alias('api_testing', 'staging'))

# alias delete karega 
# client.delete_registered_model_alias(name = 'api_testing', alias = 'production')

# ------------------- delete versions
# ye registered model k version ko delete krega
# versions = [1, 2, 3]
# for version in versions :
#      client.delete_model_version(name = 'api_testing', version = version)

# ------------- delete reg model tag
# client.delete_registered_model_tag(name = 'api_testing', key = 'tags.status')



# model load krne k liye
# model = load_model(f"models:/api_testing/2")


