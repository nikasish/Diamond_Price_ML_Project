import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
if __name__ == '__main__':
    datainj = DataIngestion()

    train_set, test_set =  datainj.initiate_data_ingestion()

    print(train_set, test_set)

    DataTransformation = DataTransformation()
    train_arr, test_arr,_= DataTransformation.initiate_data_transformation(train_set, test_set)

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)
