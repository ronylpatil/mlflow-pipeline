import pathlib
import yaml
import typing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.logger import infologger
from sklearn.base import BaseEstimator

infologger.info('*** Executing: visualize.py ***')
from src.data.make_dataset import load_data


def load_model(model_dir: str) -> BaseEstimator :
     try : 
          model = joblib.load(model_dir)
     except Exception as e : 
          infologger.info(f'exception raised while loading the model from {model_dir} [check load_model()]. exc: {e}')
     else : 
          infologger.info(f'model loaded successfully from {model_dir}')
          return model

def roc_curve() -> None : 
     pass

def conf_matrix(y_test: pd.Series, y_pred: pd.Series, labels: np.ndarray, path: pathlib.Path, params_obj: typing.IO) -> str : 
     try : 
          curr_time = datetime.now().strftime('%d%m%y-%H%M%S')
          dir_path = pathlib.Path(f'{path}/confusionMat')
          dir_path.mkdir(parents = True, exist_ok = True)
     except Exception as e : 
          infologger.info(f'there\'s an issue in directory [check conf_metrix()]. exc: {e}')
     else :
          infologger.info('directories are all set!')
          try :
               cm = confusion_matrix(y_test, y_pred, labels = labels)
               disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
               disp.plot(cmap = plt.cm.Blues)
               plt.title('Confusion Matrix')
               plt.xlabel('Predicted Label')
               plt.ylabel('True Label')
               filename = f'{dir_path.as_posix()}/{curr_time}.png'
               plt.savefig(filename)
               plt.close()
          except Exception as e : 
               infologger.info(f'there\'s some issue in ploting confusion metrix [check conf_metrix()]. exc: {e}')
          else :
               infologger.info(f'confusion metrix saved at [{dir_path}]')
               return filename
          
def main() -> None :
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent
     dir_path = f'{home_dir.as_posix()}/plots'
     # dir_path.mkdir(parents = True, exist_ok = True)

     try : 
          params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml', encoding = 'utf8'))
     except Exception as e : 
          infologger.info(f'there\'s some issue while loading params.yaml [check main()]. exc: {e}')
     else :
          data_dir = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"
          model_dir = f'{home_dir.as_posix()}{params["train_model"]["model_dir"]}/model.joblib'
          
          TARGET = params['base']['target']

          test_data = load_data(data_dir)
          x_test = test_data.drop(columns = [TARGET]).values
          y_test = test_data[TARGET]
          
          model = load_model(model_dir)
          labels = model.classes_
          try : 
               y_pred = model.predict(x_test)     # return class
          except Exception as e : 
               infologger.info(f'there\'s an issue while prediction [check main()]. exc: {e}')
          else :
               conf_matrix(y_test, y_pred, labels, dir_path, yaml_file_obj = params)
               infologger.info('program terminated normally!')
               
if __name__ == '__main__' :
     main()
