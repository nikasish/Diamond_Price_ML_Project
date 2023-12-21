import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import pickle_run
import pandas as pd

class predictpipline:
    def __init__(self):
        pass
    def predict(self, features):
        logging.info("predict function enable ... ")
        try:
            logging.info('s 1')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            logging.info('s 2')
            preprocessor = pickle_run(preprocessor_path)
            model = pickle_run(model_path)
            logging.info('s 3')

            scale_data = preprocessor.transform(features)
            y_pred = model.predict(scale_data)
            return y_pred
            
        
        except Exception as e:
            logging.info("Exception occured in Predict ... ")
            raise CustomException(e, sys)
        
class customdata:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
        logging.info("Init method completed ")

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created ... ')
            return df
        
        except Exception as e:
            logging.info('Error occured in get_data_as_dataframe ... ')
            raise CustomException(e, sys)
        