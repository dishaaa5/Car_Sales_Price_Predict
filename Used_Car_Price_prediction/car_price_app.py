import pandas as pd
import pickle
import streamlit as st
import numpy as np  


#----LOAD MODEL----
model = pickle.load(open(r"C:\Users\Dell\OneDrive\New folder\AIML(all basics)\Used_Car_Price_prediction\car_price_model.pkl" , "rb"))

st.title("Car Price Prediction Using Linear Regression")

#----INPUT FIELDS----
year = st.number_input("YEAR OF MANUFACTURE" , min_value= 1990 , max_value=2025)
km_driven = st.number_input("KILOMETERS DRIVEN" , min_value= 0 , step= 500)
fuel = st.selectbox("FUEL TYPE" , ["Petrol" , "Diesel" , "CNG" , "LPG" ," Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark dealer"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Owner Category", [
    "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"
])

#----MAP INPUTS----
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}
seller_map = {"Individual": 0, "Dealer": 1, "Trustmark dealer": 2}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {
    "First Owner": 0,
    "Second Owner": 1,
    "Third Owner": 2,
    "Fourth & Above Owner": 3,
    "Test Drive Car": 4
}

#----INPUT ARRAY----
features = np.array([[year, km_driven, fuel_map[fuel], seller_map[seller_type], trans_map[transmission], owner_map[owner]]])

# Predict
if st.button("Predict Selling Price"):
    price = model.predict(features)[0]
    st.success(f"💰 Estimated Selling Price: ₹{int(price):,}")