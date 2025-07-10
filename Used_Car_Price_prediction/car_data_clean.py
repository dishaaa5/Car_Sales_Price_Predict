import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
import pickle

#----LOAD DATA----
df = pd.read_csv(r"C:\Users\Dell\OneDrive\New folder\AIML(all basics)\Used_Car_Price_prediction\CAR DETAILS FROM CAR DEKHO (1).csv")

df.drop("name" , axis=1 , inplace=True)


#----CONVERT CATEGORICAL TO NUMERICAL----
df["fuel"] = df["fuel"].map({"Petrol" :0 , "Diesel" :1 , "CNG" :2 , "LPG" :3 , "Electric" :4})
df["seller_type"] = df["seller_type"].map({"Individual" :0 , "Dealer" :1 , "Trustmark Dealer" :2 })
df["transmission"] = df["transmission"].map({"Manual" :0 , "Automatic" :1 })
df["owner"] = df["owner"].map({"First Owner" :0 , "Second Owner" :1 , "Third Owner" :2 , "Fourth $ Above Owner" :3 , "Test Drive Car" :4})

#----FEATURES AND TARGET----
df.dropna(inplace=True)
X =  df.drop("selling_price" , axis=1) 
Y = df["selling_price"]

#----SPLIT AND TRAIN----
X_train, X_test, y_train, y_test =  train_test_split(X, Y , test_size=0.2 , random_state=42)
model = LinearRegression()
model.fit(X_train , y_train)

#----SAVE MODEL
pickle.dump(model , open("car_price_model.pkl" , "wb"))
print("Model saved as car_price_model.pkl")