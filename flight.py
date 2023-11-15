import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title('Flight Ticket Price Prediction')

st.write("""
    Welcome to the Flight Ticket Price Prediction app!

    This app aims to predict flight ticket prices using machine learning techniques based on a dataset containing various features related to flights.
    """)
st.write('## Introduction')
st.write("""
    The goal of this project is to develop a machine learning model capable of accurately predicting flight ticket prices based on features such as airline, source, distance, duration time, and more.
    """)

col1,col2,col3=st.columns(3)

with col1:
    aircraft_type=st.selectbox("Select Aircraft Type",["Airbus A320","Boeing 777","Boeing 787","Airbus A380","Boeing 737"])
    distance=st.text_input("Enter Distance KM (min:1000 & max:10000)")
    duration=st.text_input("Enter Duration Hours(min:1 & max:15)")
with col2:
    demand=st.selectbox("Select Demand",["Low","Medium","High"])
    fuel_price=st.text_input("Enter Fuel Price in $ (min:0.5 & max:1.2)")
    number_of_stops=st.selectbox("Select Number of stops",[0,1,3])
with col3:
    month_of_travel=st.selectbox("Select Month",["January","February","March","April","May","June","July","August","September","October","November","December"])
    day_of_week=st.selectbox("Select Day",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    weather_contion=st.selectbox("Select Weather Condition",["Cloudy","Snow","Rain","Clear"])
    holiday_season=st.selectbox("Select Holiday Season",["Spring","None","Fall","Summer","Winter"])



day_maping={"Wednesday":0,"Tuesday":1,"Thursday":2,"Friday":3,"Monday":4,"Saturday":5,"Sunday":6}
month_maping={"November":0,"September":1,"October":2,"March":3,"May":4,"April":5,"December":6,
               "February":7,"January":8,"August":9,"June":10,"July":11}
holiday_maping={"Winter":0,"Fall":1,"None":2,"Spring":3,"Summer":4}
demand_maping={"Low":0,"Medium":1,"High":2}
aircraft_type_maping={"Boeing 737":0,"Boeing 787":1,"Airbus A320":2,"Airbus A380":3,"Boeing 777":4}
weather_contion_maping={"Snow":0,"Rain":1,"Cloudy":2,"Clear":3}

aircraft_type=aircraft_type_maping[aircraft_type]
demand=demand_maping[demand]
month_of_travel=month_maping[month_of_travel]
day_of_week=day_maping[day_of_week]
weather_contion=weather_contion_maping[weather_contion]
holiday_season=holiday_maping[holiday_season]

if distance and duration and fuel_price:
    distance=int(distance)
    duration=float(duration)
    fuel_price=float(fuel_price)
    
    import pickle
    
    with open("C:\\Users\\santh\\OneDrive\\Documents\\flight_model.pkl","rb") as file:
        model=pickle.load(file)
    new_sample=np.array([[distance,duration,aircraft_type,int(number_of_stops),int(day_of_week),int(month_of_travel),
                          int(holiday_season),int(demand),int(weather_contion),fuel_price]])
    new_pred=model.predict(new_sample)
    if st.button("Price"):
        st.write("## :green[Price]",new_pred)
    










