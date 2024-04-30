import mlflow
import streamlit as st
from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model

st.set_page_config(page_title = 'WineQ Prediction',
                    page_icon = '🦅', 
                    layout = 'centered',
                    initial_sidebar_state = 'expanded') 

# location where MLflow is storing experiment data, such as metrics, parameters, models, and artifacts
# MLFLOW_TRACKING_URI = params['mlflow_config']['mlflow_tracking_uri']
# you set the tracking URI globally using mlflow.set_tracking_uri(). When you create the MlflowClient object
#  without specifying the tracking_uri parameter explicitly, it automatically uses the tracking URI that you've
#  set globally. This approach is more concise. 
mlflow.set_tracking_uri("mysql+pymysql://admin:Admin123@mysqldb.chqgugyi6uuj.us-east-1.rds.amazonaws.com:3306/mysql_mlflow")
client = MlflowClient()

# Sidebar Info
st.sidebar.title("About Me 🤖")
try :
     model = client.get_model_version_by_alias(name = 'outperforming models', alias = 'production')
     
     st.sidebar.write(f"#### Model Name\n ```{model.name}```")
     st.sidebar.write(f"#### Model Version\n ```version v{model.version}```")
     st.sidebar.write(f"#### Current Stage")
     for i in model.aliases: 
          st.sidebar.write(f'```{i}```')
     st.sidebar.write(f"#### Run ID\n ```{model.run_id}```")

     if 'final_model' not in st.session_state :        
          with st.spinner('Loading Model') : 
               # loading model using version, achieve same using alias
               st.session_state['final_model'] = load_model(f"models:/{model.name}@{'production'}")
     st.sidebar.info('##### Server is Up 🔥')
except :
     st.sidebar.warning('##### ⚠️ Model not found')

#  Main Area   
st.title("Wine Quality 🍷")    
fixed_acidity = st.number_input('Fixed Acidity *',min_value = 4.1, max_value = 16.4, value = None, placeholder = '4.1 <= Fixed Acidity <= 16.4', step = 0.1)
volatile_acidity = st.number_input('Volatile Acidity *', placeholder = '0.5 <= Volatile Acidity <= 1.98', min_value = 0.5, max_value = 1.98, value = None, step = 0.1)
citric_acid = st.number_input('Critic Acid *', placeholder = '0.0 <= Citrix Acid <= 1.5', min_value = 0.0, max_value = 1.5, value = None, step = 0.1)
residual_sugar = st.number_input('Residual Sugar *', placeholder = '0.5 <= Residual Sugar <= 16.0', min_value = 0.5, max_value = 16.0, value = None, step = 0.1)
chlorides = st.number_input('Chloride *', placeholder = '0.008 <= Chloride <= 0.7', min_value = 0.008, max_value = 0.7, value = None, step = 0.001, format = "%.3f")
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide *', placeholder = '0.7 <= Free Sulfur Dioxide <= 70.0', min_value = 0.7, max_value = 70.0, value = None, step = 0.1)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide *', placeholder = '5 <= Total Sulfur Dioxide <= 290', min_value = 5, max_value = 290, value = None, step = 4)
density = st.number_input('Density *', placeholder = '0.85 <= Density <= 1.5', min_value = 0.85, max_value = 1.5, value = None, step = 0.1)
pH = st.number_input('PH *', placeholder = '2.6 <= PH <= 4.5', min_value = 2.6, max_value = 4.5, value = None, step = 0.1)
sulphates = st.number_input('Sulphate *', placeholder = '0.2 <= Sulphate <= 2.5', min_value = 0.2, max_value = 2.5, value = None, step = 0.1)
alcohol = st.number_input('Alcohol *', placeholder = '8 <= Alcohol <= 15', min_value = 8, max_value = 15, value = None, step = 1)

user_input = {'fixed_acidity': fixed_acidity, 'volatile_acidity' : volatile_acidity, 'citric_acid': citric_acid, 'residual_sugar' : residual_sugar,
              'chlorides': chlorides, 'free_sulfur_dioxide': free_sulfur_dioxide, 'total_sulfur_dioxide': total_sulfur_dioxide, 'density': density, 
               'pH': pH, 'sulphates': sulphates, 'alcohol': alcohol}

if fixed_acidity and volatile_acidity and citric_acid and residual_sugar and chlorides and free_sulfur_dioxide and total_sulfur_dioxide and density and\
          pH and sulphates and alcohol :
     # feature engineering
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
     
     input_data =   [[user_input['fixed_acidity'], user_input['volatile_acidity'], user_input['citric_acid'], user_input['residual_sugar'],
                      user_input['chlorides'], user_input['free_sulfur_dioxide'], user_input['total_sulfur_dioxide'], user_input['density'], 
                      user_input['pH'], user_input['sulphates'], user_input['alcohol'], user_input['total_acidity'], user_input['acidity_to_pH_ratio'], 
                      user_input['free_sulfur_dioxide_to_total_sulfur_dioxide_ratio'], user_input['alcohol_to_acidity_ratio'], 
                      user_input['residual_sugar_to_citric_acid_ratio'], user_input['alcohol_to_density_ratio'], user_input['total_alkalinity'], 
                      user_input['total_minerals']]]

     if st.button('Predict ⚙️') :
          with st.spinner('Processing data...') :
               op = st.session_state['final_model'].predict(input_data)[0]
               pred_prob = st.session_state['final_model'].predict_proba(input_data)[0]
               st.success(f'Predicted Quality is {op} with {(max(pred_prob) * 100):.2f}% confidence.')

# cmd: streamlit run ./prod/client.py
               