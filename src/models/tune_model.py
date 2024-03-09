import mlflow
import yaml
import pathlib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from src.data.make_dataset import load_data
from functools import partial
from src.logger import infologger

infologger.info('*** Executing: tune_model.py ***')
from src.visualization import visualize

def objective(params: dict, yaml_obj: dict, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, plots_dir: str) -> dict :
     try : 
          mlflow_config = yaml_obj['mlflow_config']
          remote_server_uri = mlflow_config['remote_server_uri']
          exp_name = mlflow_config['tunningExpName']
          mlflow.set_tracking_uri(remote_server_uri)
          mlflow.set_experiment(experiment_name = exp_name)
          # adding experiment description
          experiment_description = ('optimizing the hyperparameters of a machine learning model using Hyperopt') 
          mlflow.set_experiment_tag("mlflow.note.content", experiment_description)
     except Exception as e : 
          infologger.info(f'exception occured while intializing mlflow exp [check objective()]. exc: {e}')
     else :
          try : 
               model = RandomForestClassifier(**params)
               model.fit(x_train, y_train)
               y_pred = model.predict(x_test)
               y_pred_prob = model.predict_proba(x_test)
               
               accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
               precision = metrics.precision_score(y_test, y_pred, zero_division = 1, average = 'macro')
               recall = metrics.recall_score(y_test, y_pred, average = 'macro')
               roc_score = metrics.roc_auc_score(y_test, y_pred_prob, average = 'macro', multi_class = 'ovr')

               with mlflow.start_run(description = 'tunning RFC also using hyperopt optimization technique') :
                    mlflow.set_tags({'project_name': 'wine-quality', 'author' : 'ronil', 'project_quarter': 'Q1-2024'})
                    mlflow.log_params(params)
                    filename = visualize.conf_matrix(y_test, y_pred, model.classes_, path = plots_dir, params_obj = yaml_obj)
                    mlflow.log_artifact(filename, 'confusion_matrix')
                    mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "roc_score": roc_score})
                    mlflow.sklearn.log_model(model, 'model')
          except Exception as e :
               infologger.info(f'got exception while tracking exeriments [check objective()]. exc: {e}')
          else :
               return {'loss': -accuracy, 'status': STATUS_OK}

def main() -> None :
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent
     cm_dir = f'{home_dir.as_posix()}/plots/tunning'

     params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml'))
     parameters = params['train_model']
     TARGET = params['base']['target']
     train_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_train.csv"
     test_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"

     train_data = load_data(train_data)
     x_train = train_data.drop(columns = [TARGET]).values
     y_train = train_data[TARGET]

     test_data = load_data(test_data)
     x_test = test_data.drop(columns = [TARGET]).values
     y_test = test_data[TARGET]

     # hyperopt
     additional_params = {'yaml_obj': params, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                         'plots_dir': cm_dir}

     partial_obj = partial(objective, **additional_params)

     # we can take the range as input via params.yaml
     search_space = {'n_estimators': hp.randint('n_estimators', 200 - 15) + 15,
                     'criterion': hp.choice('criterion', ['gini', 'entropy']),
                     'max_depth': hp.randint('max_depth', 100 - 5) + 5,
                     'min_samples_split': hp.randint('min_samples_split', 100 - 5) + 5,
                     'min_samples_leaf': hp.randint('min_samples_leaf', 100 - 10) + 10 }
     try : 
          best_result = fmin(fn = partial_obj,
                              space = search_space,
                              algo = tpe.suggest,
                              max_evals = params['hyperopt']['max_eval'],
                              trials = Trials())

     except Exception as e :
          infologger.info(f'exception raised while tunning model using hyperopt [check main()]. exc: {e}')
     else :
          infologger.info('program terminated normally!')

if __name__ == '__main__' : 
     main()
     infologger.info('tune_model.py as __main__')



# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host localhost -p 5000

# UserWarning: Distutils was imported before Setuptools, but importing Setuptools also replaces the `distutils` module
               #  in `sys.modules`. This may lead to undesirable behaviors or errors. To avoid these issues, avoid 
               #  using distutils directly, ensure that setuptools is installed in the traditional way (e.g. not an 
               # editable install), and/or make sure that setuptools is always imported before distutils.
# Solution : So I simply deleted the _distutils_hack and distutils-precedence.pth from the site-packages directory.
             # So far so good, though ymmv! My best guess is that those are left behind from some older version of 
             # setuptools and are not removed when setuptools is updated.
