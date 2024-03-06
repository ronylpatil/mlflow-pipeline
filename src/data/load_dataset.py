import pathlib
import yaml
import pandas as pd
from src.logger import infologger

infologger.info('*** Executing: load_dataset.py ***')

# load data from given path and return df
def load_data(remote_loc: str) -> pd.DataFrame : 
     try : 
          # correct way to read data from drive
          remote_loc = 'https://drive.google.com/uc?id=' + remote_loc.split('/')[-2]  
          df = pd.read_csv(remote_loc)  
     except Exception as e : 
          infologger.info(f'there\'s some issue while loading data from remote server [check load_data()]. exc: {e}')
     else : 
          infologger.info(f'data loaded from {remote_loc}')
          return df

# save data at data/raw dir
def save_data(data: pd.DateOffset, output_path: str, file_name: str) -> None : 
     try : 
          data.to_csv(path_or_buf = output_path + f'/{file_name}.csv', index = False)
     except Exception as e : 
          infologger.info(f'there\'s some issue while saving the data [check save_data()]. exc: {e}')
     else : 
          infologger.info(f'data saved at [path: {output_path}/{file_name}.csv]')

# load data & then save it
def main() -> None : 
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent

     params_file = home_dir.as_posix() + '/params.yaml'
     try : 
          params = yaml.safe_load(open(params_file))
     except Exception as e : 
          infologger.info(f'there\'s some issue while loading the params file or output path [check main()]. exc: {e}')
     else : 
          # create dir if not present, else execute without any warning/error
          output_path = home_dir.as_posix() + params['load_dataset']['raw_data']
          pathlib.Path(output_path).mkdir(parents = True, exist_ok = True)
          data = load_data(params['load_dataset']['drive_link'])
          save_data(data, output_path = output_path, file_name = params['load_dataset']['file_name'])
          infologger.info('program terminated normally!')

if __name__ == "__main__" : 
     infologger.info('load_dataset.py as __main__')
     main()
