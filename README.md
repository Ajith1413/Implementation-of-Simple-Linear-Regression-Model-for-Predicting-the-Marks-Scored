# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the necessary libraries and read the file student scores
2. print the x and y values
3. separate the independent values and dependent values
4. split the data
5. create a regression model
6. find mse,mae and rmse and print the values
   
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AJITH KUMAR A
RegisterNumber:  23002150
*/
```
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
df.tail()
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='purple')
plt.plot(x_test,regressor.predict(x_test),color='yellow')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
~~~

## Output:
![image](https://github.com/RakshithaK11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139336455/2149c2cd-28d4-4e91-b592-a6f5b9455ff3)
![image](https://github.com/RakshithaK11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139336455/e5cec305-c4d7-42da-b1d3-d2f93a7056cb)
![image](https://github.com/RakshithaK11/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139336455/ad3e916d-238a-417e-8280-13323d59f14d)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
