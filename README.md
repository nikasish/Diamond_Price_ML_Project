### I am very much happy to share with you my first ML project.  I completed my first ML project on Diamond Price Prediction. Basically it is regression type of problem; here we predict the diamond price based on the various features of diamond. 

#### this is not cloud deployed project..... 

### Steps I followed to complete the project:- 
### 1.	*Data Ingestion*
### 2.	Load data in my jupyter notebook and read the data by pandas.
### 3.	Performing EDA on it like (information of data, how many rows and columns and also how many categorical and numerical values, have any missing, duplicated value, checking the co-relation and many more stastical operations ) to understand the data.
### 4.	Performing Preprocessing (creating the pipeline, to handle the missing value, encoding the categorical value, Scale down the data and many more things we have to do, based on the use cases).
### 5.	Work on Data Ingestion Components.
### 6.	Work on Data Transformation Components.
### 7.	Work on Model Training Components.
### 8.	Combine {steps (5, 7, 8)} components into a single training pipeline components  and start training the model. 
### 9.	After completing the model training model and scalar pickle file crated and with this we can predict the new data. 
### 10.	Working with UI in flask api (application.py) file.
### 11.	Everything set we are ready to predict into UI.

### I tried with multiple regression algorithm but my best algorithm is Random Forest ()

### Random Forest: - Random forest is an ensemble technique. Whenever we give data to the random forest algorithm internally random forest create n no’s of base learner and passed the sample of row data and sample of features data to all the base learners yes data may be repeated there is no problem. After that all base learner output will combine and based learner output will select by the majority voting. (Here in our case we are solving the regression problem so we pick all the output and find the average of the output) and this output will our random forest final output.

#### Note: - base learner is nothing is a decision tree. In classification we pick all the base learners output and select majority voting output as random forest output and in regression we simple calculate the average of base learner output. 

## to clone this project copy the url from url bar, open your folder and write git clone <paste url> (copied url from url bar or address bar )
## after the few sec entire project clone in your local syatem.

# To find the prediction:- 
## 1. create a virtual enviroment in your local system 
## 2. run the setup.py file  by <python setup.py install> 
## if the requirement.txt file is not run plz run the requirement.txt file by <python -r requirement.txt>
## 3. run the training pipeline component by <python src/pipelines/training_pipeline.py>
## 4. then run the <application.py> file 
### now you are ready to find the prediction. 

#### Thanking you 
