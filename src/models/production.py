'''
Plan of Action
1. Search an efficient model from experiment which is having best metrics in its territory and get the details of it
2. check given register model is new or existing, 
     if new :
          - Register that model
          - Add description and assign some tags to registered model
          - Add description to the latest version along with tags and aliases
     if already exist :
          - get the details of production model
          - compare the performance metrics of old model and new model
               - if new model outperforming
                    - register that model & put it into production env
               else:
                    - don't do anything, as high performing model already in production
'''

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

exp_name = 'modeltunning'

exp_id = client.search_experiments(filter_string = f"name = '{exp_name}'")[0].experiment_id
new_model = client.search_runs(experiment_ids = exp_id, order_by = ['metrics.accuracy DESC', 'metrics.precision DESC', 'metrics.recall DESC'])[4]
run_id = new_model.info.run_id
reg_model_name = 'model2'

if client.search_registered_models(filter_string = f"name = '{reg_model_name}'") :     # model already register he
     try : 
          prod_version = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').version
     except MlflowException as e : 
          print('Exception: ', e)
     else :
          prod_run_id = client.get_model_version_by_alias(name = reg_model_name, alias = 'production').run_id
          prod_accuracy = mlflow.get_run(run_id = prod_run_id).data.metrics['accuracy']

          if new_model.data.metrics['accuracy'] > prod_accuracy :
               model_uri = f"runs:/{run_id}/sklearn-model"
               mv = mlflow.register_model(model_uri, name = reg_model_name, tags = {'val_status': 'approved'})

               client.update_registered_model(name = reg_model_name, description = 'Filtered models from HYPEROPT tunned outperforming models')
               client.set_registered_model_tag(name = reg_model_name, key = 'developer', value = 'ronil')
               client.update_model_version(name = reg_model_name, version = mv.version, description = 'outperforming model')
               client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production')
               client.set_registered_model_alias(name = reg_model_name, version = prod_version, alias = 'archive')
               client.set_model_version_tag(name = reg_model_name, version = prod_version, key = 'val_status', value = 'trash')
          else : 
               print('High performing model already in production.')
else :     # model register hi nahi he 
     model_uri = f"runs:/{run_id}/sklearn-model"
     mv = mlflow.register_model(model_uri, name = reg_model_name, tags = {'val_status': 'approved'})
     client.update_registered_model(name = reg_model_name, description = 'Filtered models from HYPEROPT tunned outperforming models')
     client.set_registered_model_tag(name = reg_model_name, key = 'developer', value = 'ronil')
     client.update_model_version(name = reg_model_name, version = mv.version, description = 'outperforming model')
     client.set_registered_model_alias(name = reg_model_name, version = mv.version, alias = 'production')

# we can do whatever we want, it's very flexible