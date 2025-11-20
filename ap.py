import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
#loading the csv data to a Pandas DataFrane
heart_data = pd.read_csv('heart_disease_data (1).csv')
#print first 5 row
heart_data.head()
# print last 5 row
heart_data.tail()
#number of rows and col
heart_data.shape
#getting some info about data
heart_data.info()
#checking for missing values
heart_data.isnull().sum()
#stastistical measures about the data
heart_data.describe()
#checking distribution of target variables
heart_data['target'].value_counts()
x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x.shape,x_train.shape,x_test.shape)
model = LogisticRegression()
#training the lr model with traning data
model.fit(x_train,y_train)
#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('accuracy on training data:',training_data_accuracy)
# accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy on test data:',test_data_accuracy)
input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,0)
#change the input data to a numpy array 
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
    print('the person does not have a heart disease')
else:
    print('the person has heart disease')

# Save the trained model
pickle.dump(model, open('heart_disease_model.sav', 'wb'))
