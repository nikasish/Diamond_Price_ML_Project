import os, sys
import pandas as pd
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from src.utils import model_selection

@dataclass
class ModelTrainerConfig:
    model_pickle_file = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_arr, test_arr):
        logging.info("initate_model_training started .... ")
        try:
            logging.info("selecting the Xtrain Xtest ytrain ytest ... ")
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]

            )

            logging.info("model implemented ... ")
            models = {
                'LinearRegression' : LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'ElasticNet' : ElasticNet(),
                'SVR' : SVR(),
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'AdaBoostRegressor' : AdaBoostRegressor(),
                'RandomForestRegressor' : RandomForestRegressor(),
                'GradientBoostingRegressor' : GradientBoostingRegressor() 
                }
            
            model_report:dict=model_selection(X_train, X_test, y_train, y_test, models)
            print(model_report)
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found : {best_model_name} , R2 Score : {best_model_score}')
            logging.info(f'Best Model Found : {best_model_name} , and R2 Score is : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.model_pickle_file,
                obj=best_model
            )

        except Exception as e:
            logging.info( "Error occured in initate_model_training ")

            raise CustomException(e, sys)

