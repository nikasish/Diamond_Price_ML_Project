import os, sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object


## Datatransformationconfig class
@dataclass
class DataTransformationconfig:
    transformar = os.path.join('artifacts', 'preprocessor.pkl')

## Datatransformation class 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()


    def get_data_transformation_object(self):
        logging.info("get_data_transformation_object started ... ")

        try:
            logging.info(" categories the data into numerical and categorical ... ")
            numerical_col = ['carat', 'depth','table', 'x', 'y', 'z']
            cat_col = ['cut', 'color','clarity']

            logging.info('preserve the the order for ordinal data ... ')
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('initiated the pipeline ... ')
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('o_encoding', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
            ])

            logging.info("column Transformar Initiated ... ")
            transformar = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, cat_col),
                ('numerical_pipeline', numerical_pipeline, numerical_col)
            ])

            logging.info('pipeline completed and return ... ')
            
            return transformar
        
        except Exception as e:
            logging.info(" error occured in get_data_transformation_object ... ")
        


    def initiate_data_transformation(self, train_set, test_set):
        logging.info("initiate_data_transformation Started ... ")

        try:
            logging.info('data reading by pandas ... ')
            train_set= pd.read_csv(train_set)
            test_set = pd.read_csv(test_set)
            logging.info(f'train dataset : \n {train_set.head().to_string()}')
            logging.info(f'test dataset : \n {test_set.head().to_string()}')

            logging.info("column transformation object called ... ")
            ColumnTransformer_obj = self.get_data_transformation_object()
    

            logging.info("selecting target and input data ... ")
            y = 'price'
            X = [y, 'id']

            X_train = train_set.drop(columns=X, axis=1)
            y_train = train_set[y]

            X_test = test_set.drop(columns=X, axis=1)
            y_test = test_set[y]

            logging.info('scalign the input and target data ... ')
            X_train_scale = ColumnTransformer_obj.fit_transform(X_train)
            X_test_scale = ColumnTransformer_obj.transform(X_test)

            logging.info('concatinate the train and test data ... ')

            train_arr = np.c_[X_train_scale, np.array(y_train)]
            test_arr = np.c_[X_test_scale, np.array(y_test)]

            save_object(
                file_path=self.data_transformation_config.transformar,
                obj=ColumnTransformer_obj
            )

            logging.info(" pickle file created and saved to artifacts folder ... ")

            logging.info("now we will return the final dataset ... ")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.transformar
            )

        except Exception as e :
            logging.info("error occured in initiate_data_transformation")
            raise CustomException(e, sys)

