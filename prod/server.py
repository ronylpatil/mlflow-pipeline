import mlflow
import pathlib
import time
import yaml
import webbrowser
import typing
import subprocess as sp
import streamlit as st
from mlflow.sklearn import log_model
from src.data.make_dataset import load_data
from src.models.train_model import train_model
from src.visualization import visualize
from src.models.train_model import save_model

def train(n_est: int, crit: str, maxd: int, mss: int, msl: int, params: typing.Dict) -> tuple : 
     # curr_dir = pathlib.Path(__file__)
     # home_dir = curr_dir.parent.parent
     # params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml', encoding = 'utf8'))
     plots_dir = f'{home_dir.as_posix()}/plots/training_plots'

     parameters = params['train_model']
     TARGET = params['base']['target']

     train_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_train.csv"
     model_dir = f"{home_dir.as_posix()}{parameters['model_dir']}"
     pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
     
     data = load_data(train_data)
     X_train = data.drop(columns = [TARGET]).values
     Y = data[TARGET]

     outs = train_model(X_train, Y, n_estimators = n_est, criterion = crit, max_depth = maxd, min_samples_split = mss, min_samples_leaf = msl, random_state = parameters['random_state'] , yaml_file_obj = params)
     
     filename = visualize.conf_matrix(Y, outs['y_pred'], labels = outs['model'].classes_, path = plots_dir, params_obj = params)

     return outs, filename, model_dir, params

def new_experiment(exp_name: str, exp_des: str, run_desc: str, n_est: int, crit: str, maxd: int, mss: int, msl: int, params: typing.Dict) -> float : 
     outs, filename, model_dir, params = train(n_est, crit, maxd, mss, msl, params)

     mlflow_config = params['mlflow_config']
     remote_server_uri = mlflow_config['remote_server_uri']

     mlflow.set_tracking_uri(remote_server_uri)
     mlflow.set_experiment(experiment_name = exp_name)
     # adding experiment description    
     mlflow.set_experiment_tag("mlflow.note.content", exp_des)

     # runs description
     with mlflow.start_run(description = run_desc) : 
          mlflow.log_params({"n_estimator": outs['params']['n_estimator'], "criterion": outs['params']['criterion'], 
                              "max_depth": outs['params']['max_depth'], "random_state": outs['params']['seed']})
          mlflow.log_metrics({"accuracy": outs['metrics']['accuracy'], "precision": outs['metrics']['precision'], 
                              "recall": outs['metrics']['recall'], "roc_score": outs['metrics']['roc_score']})
          log_model(outs['model'], "model")
          mlflow.log_artifact(filename, 'confusion_matrix')
          mlflow.set_tags({'project_name': 'wine-quality', 'project_quarter': 'Q1-2024'})

     save_model(outs['model'], model_dir)
     return outs['metrics']['accuracy']

def existing_exp(exp_name, run_desc, n_est, crit, maxd, mss, msl, params) -> float :
     outs, filename, model_dir, params = train(n_est, crit, maxd, mss, msl, params)

     mlflow_config = params['mlflow_config']
     remote_server_uri = mlflow_config['remote_server_uri']

     mlflow.set_tracking_uri(remote_server_uri)
     mlflow.set_experiment(experiment_name = exp_name)

     # runs description
     with mlflow.start_run(description = run_desc) : 
          mlflow.log_params({"n_estimator": outs['params']['n_estimator'], "criterion": outs['params']['criterion'], 
                              "max_depth": outs['params']['max_depth'], "random_state": outs['params']['seed']})
          mlflow.log_metrics({"accuracy": outs['metrics']['accuracy'], "precision": outs['metrics']['precision'], 
                              "recall": outs['metrics']['recall'], "roc_score": outs['metrics']['roc_score']})
          log_model(outs['model'], "model")
          mlflow.log_artifact(filename, 'confusion_matrix')
          mlflow.set_tags({'project_name': 'wine-quality', 'project_quarter': 'Q1-2024'})

     save_model(outs['model'], model_dir)
     return outs['metrics']['accuracy']

def open_mlflow_ui(cmd) -> None :
#     cmd = "mlflow ui --port 5050"
#     cmd = 'mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host localhost -p 5000'
    sp.Popen(cmd, shell = True)

def open_browser(url) -> None :
    webbrowser.open_new_tab(url)


curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent
params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml', encoding = 'utf8'))


st.set_page_config(page_title = 'WineQ Model Tuner',
    page_icon = '🦅', 
    layout = 'centered',
    initial_sidebar_state = 'expanded') 

# Sidebar for hyperparameter tuning
st.sidebar.title('Params Tuner ⚙️')
n = st.sidebar.slider('N-Estimators', min_value = 1, max_value = 250, step = 2, value = 25)
d = st.sidebar.slider('Max Depth', min_value = 1, max_value = 50, step = 2, value = 5)
mss = st.sidebar.slider('Min Samples Split', min_value = 20, max_value = 100, step = 2, value = 35)
msl = st.sidebar.slider('Min Samples Leaf', min_value = 10, max_value = 100, step = 2, value = 25)
c = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'], index = 1)

st.sidebar.title('Mlflow Tracking 🔎')  
if st.sidebar.button('Launch 🚀') :
     remote_server_uri = params['mlflow_config']['remote_server_uri']
     open_mlflow_ui(params['mlflow_config']['cmd'])
     st.sidebar.success(f'Server is live [here]({remote_server_uri})', icon = '🔥')
     #     open_browser('http://localhost:5000')
     open_browser(remote_server_uri)

# Main Page Content
st.title('WineQ Prediction Model Trainer')
exp_type = st.radio('Select Experiment Type', ['New Experiment', 'Existing Experiment'], horizontal = True)
if exp_type == 'New Experiment' :
     exp_name = st.text_input('Enter Experiment Name *', placeholder = 'Experiment Name')
     exp_des = st.text_area('Enter Experiment Description *', placeholder = 'Description')
     run_desc = st.text_input('Enter Run\'s Description *', placeholder = 'Run\'s Description')
else :
     try :
          MLFLOW_TRACKING_URI = params['mlflow_config']['mlflow_tracking_uri']
          mlflow.set_tracking_uri(uri = MLFLOW_TRACKING_URI)
          exps = [i.name for i in mlflow.search_experiments()]
          if pathlib.Path('./mlruns').exists() :
               exp_name = st.selectbox('Select Experiment', exps, index = None, placeholder = 'Choose Experiment')
               run_desc = st.text_input('Enter Run\'s Description', placeholder = 'Run\'s Description')
          else:
               st.warning('🚨 No Previous Experiments Found! Create New Experiment ⬆️')            
     except:
          st.warning('🚨 No Previous Experiments Found! Create New Experiment ⬆️')

# Training the model starts from here    
if st.button('Train ⚙️') :
     if exp_type == 'New Experiment' and exp_name and exp_des and run_desc :
          with st.spinner('Training the model...') :
               acc = new_experiment(exp_name, exp_des, run_desc, n_est = n, crit = c, maxd = d, mss = mss, msl = msl, params = params)
          st.success(f'Training Accuracy Achieved {(acc * 100):.3f}%')
     elif exp_type == 'Existing Experiment' and exp_name and run_desc : 
          with st.spinner('Training the model...') :
               acc = existing_exp(exp_name, run_desc, n_est = n, crit = c, maxd = d, mss = mss, msl = msl, params = params)
          st.success(f'Training Accuracy Achieved {(acc * 100):.3f}%') 
     else : 
          st.warning('Please ensure all obligatory fields are filled') 
     

# cmd: streamlit run ./production/train_app.py