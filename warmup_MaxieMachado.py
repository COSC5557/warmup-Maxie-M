## Description: I decided to use the red wine dataset provided for this exercise. 
## I imported necessary tools to be able to use my chosen regression and
## classification models. I moved on to reading the dataset, naming it 
## “red_wine_data”, then proceeded to print out the first and last five lines of the 
## data set. Once this has been completed I split my data into train and test sets, 
## going on to train the data. Once this is completed I use Linear Regression on my 
## dataset, printing out the accuracy of Linear regression. 
## For this the score = 0.363257630212877, which is not very high. Lastly I moved 
## on to doing Logistic Regression, creating a scaler to properly be able to use this 
## model. Once this is completed we use Logistic Regression, and then print out the 
## mean absolute error. Which the mean absolute error = 40.0, which is better then 
## linear regressions score. 

## Resources I used to reference: 
## https://www.youtube.com/watch?v=A2zlm3NkeDk
## https://www.youtube.com/watch?v=iehavRcmPNY&t=46s
## https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/
## https://www.simplilearn.com/tutorials/scikit-learn-tutorial/sklearn-linear-regression-with-examples
## https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods
## https://medium.datadriveninvestor.com/regression-from-scratch-wine-quality-prediction-d61195cb91c8 
## https://towardsdatascience.com/red-wine-quality-prediction-using-regression-modeling-and-machine-learning-7a3e2c3e1f46

#importing needed tools 
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 

#reading RED wine data, printing first 5 rows, checking type of data 
red_wine_data = pd.read_csv("winequality-red.csv")
red_wine_data.head() 
red_wine_data.info()

#splitting into train and test   
X = red_wine_data.drop('quality' , axis = 1)
y = red_wine_data['quality']
red_wine_data.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Linear Regression 
red_wine_data_linear = LinearRegression().fit(X_train , y_train) 
linear_score = red_wine_data_linear.score(X_train , y_train)

print('Score: ', linear_score)

#Logistic Regression 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

red_wine_data_logic = LogisticRegression() 
red_wine_data_logic.fit(X_train_scaled , y_train) 

y_pred_logic = red_wine_data_logic.predict(X_test_scaled)

mae = (mean_absolute_error(y_test , y_pred_logic) *100)

print('Mean Absolute Error: ' , mae)
