# build api that will show model details and prediction
import pathlib
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Define your FastAPI app
app = FastAPI()

# Define request body model using Pydantic
class WineqIp(BaseModel) : 
     fixed_acidity : float = Field(..., ge = 4.1, le = 16.4)
     volatile_acidity : float = Field(..., ge = 0.5, le = 1.98)  
     citric_acid : float = Field(..., ge = 0, le = 1.5)
     residual_sugar : float = Field(..., ge = 0.5, le = 16)
     chlorides : float = Field(..., ge = 0.008, le = 0.7)
     free_sulfur_dioxide : float = Field(..., ge = 0.7, le = 70)
     total_sulfur_dioxide : float = Field(..., ge = 5, le = 290)
     density : float = Field(..., ge = 0.85, le = 1.5)
     pH : float = Field(..., ge = 2.6, le = 4.5)
     sulphates : float = Field(..., ge = 0.2, le = 2.5)
     alcohol : float = Field(..., ge = 8, le = 15)

# Define your machine learning model class
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent
params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml'))
mlflow.set_tracking_uri(params['mlflow_config']['mlflow_tracking_uri'])
client = MlflowClient()
# fetch model by model version
model_details = client.get_model_version_by_alias(name = params['mlflow_config']['reg_model_name'], alias = params['mlflow_config']['stage'])
# model = load_model(f'models:/{model_details.name}/{model_details.version}')
# fetch model by model alias
model = load_model(f"models:/{model_details.name}@{params['mlflow_config']['stage']}")

def feat_gen(user_input: dict) -> dict : 
     user_input['total_acidity'] = user_input['fixed_acidity'] + user_input['volatile_acidity'] + user_input['citric_acid']

     user_input['acidity_to_pH_ratio'] = (lambda total_acidity, pH : 0 if pH == 0 else total_acidity / pH)(user_input['total_acidity'], user_input['pH'])

     user_input['free_sulfur_dioxide_to_total_sulfur_dioxide_ratio'] = (lambda free_sulfur_dioxide, total_sulfur_dioxide : 0 if total_sulfur_dioxide == 0 \
                                                                           else free_sulfur_dioxide / total_sulfur_dioxide)\
                                                                           (user_input['free_sulfur_dioxide'], user_input['total_sulfur_dioxide'])

     user_input['alcohol_to_acidity_ratio'] = (lambda alcohol, total_acidity : 0 if total_acidity == 0 else alcohol / total_acidity)\
                                                  (user_input['alcohol'], user_input['total_acidity'])

     user_input['residual_sugar_to_citric_acid_ratio'] = (lambda residual_sugar, citric_acid : 0 if citric_acid == 0 else residual_sugar / citric_acid)\
                                                            (user_input['residual_sugar'], user_input['citric_acid'])

     user_input['alcohol_to_density_ratio'] = (lambda alcohol, density : 0 if density == 0 else alcohol / density)(user_input['alcohol'], user_input['density'])
     user_input['total_alkalinity'] = user_input['pH'] + user_input['alcohol']
     user_input['total_minerals'] = user_input['chlorides'] + user_input['sulphates'] + user_input['residual_sugar']

     return user_input

@app.get('/')
def root() :
     return {'api status': 'up & running'}

# Define API route to make predictions
@app.post('/predict')
def predict_wineq(input_data: WineqIp) :
     processed_data = feat_gen(input_data.model_dump())

     data =   [[processed_data['fixed_acidity'], processed_data['volatile_acidity'], processed_data['citric_acid'], processed_data['residual_sugar'],
                    processed_data['chlorides'], processed_data['free_sulfur_dioxide'], processed_data['total_sulfur_dioxide'], processed_data['density'], 
                    processed_data['pH'], processed_data['sulphates'], processed_data['alcohol'], processed_data['total_acidity'], processed_data['acidity_to_pH_ratio'], 
                    processed_data['free_sulfur_dioxide_to_total_sulfur_dioxide_ratio'], processed_data['alcohol_to_acidity_ratio'], 
                    processed_data['residual_sugar_to_citric_acid_ratio'], processed_data['alcohol_to_density_ratio'], processed_data['total_alkalinity'], 
                    processed_data['total_minerals']]]

     # here we can not pass this predictions output directly as api output because it's numpy array ok...
     # and numpy array is not compatible with any web framework
     # I stuck here for 3 hrs, this error was damn...
     predictions = model.predict(data).tolist()
     pred_probability = [round(i, 5) for i in model.predict_proba(data).tolist()[0]]
     return {'predictions': predictions, 'probability': pred_probability}

@app.get('/details')
def get_details() :
     return {'model name': model_details.name, 'model version': model_details.version, 
             'model aliases': model_details.aliases, 'model run_id': model_details.run_id,
             'model description': model_details.description, 'model tags': model_details.tags}

if __name__ == '__main__' :
     import uvicorn
     uvicorn.run('prod.api:app', host = '127.0.0.1', port = 8000, log_level = 'debug',
                    proxy_headers = True, reload = True)

# app_name: api (file name)
# port: 8000 (default)
# cmd: uvicorn prod.api:app --reload
