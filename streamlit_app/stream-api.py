import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data=pd.read_csv("cleaned_data.csv")
scaler.fit(data)
x = scaler.transform(data)

pickle_in = open("model_test.pkl","rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "LiveStock Disease Detection App"

def predict_mastitis(Breed,Months,History,IUFL,EUFL,IUFR,EUFR,IURL,EURL,IURR,EURR,Temperature,Hardness,Pain):
    array = [Breed,Months,History,IUFL,EUFL,IUFR,EUFR,IURL,EURL,IURR,EURR,Temperature,Hardness,Pain]
    array=scaler.transform(np.expand_dims(array, axis=0))
    prediction = classifier.predict(array)
    return prediction




def main():
    st.title("LiveStock Disease Detection App")
    html_temp="""
    <div style ="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Mastitis Detector </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Breed = st.selectbox("Breed:",('Jersey','Hostlene'),key=14)
    if (Breed == 'Jersey'):
        Breed = 0
    else: 
        Breed = 1
    Months = st.text_input("Months after giving birth:",key=0)
    History = st.text_input("Previous Mastitis:",key=1)
    IUFL = st.text_input("IUFL:",key=2)
    EUFL = st.text_input("EUFL:",key=3)
    IUFR = st.text_input("IUFR:",key=4)
    EUFR = st.text_input("EUFR:",key=5)
    IURL = st.text_input("IURL:",key=6)
    EURL = st.text_input("EURL:",key=9)
    IURR = st.text_input("IURR:",key=10)
    EURR = st.text_input("EURR:",key=11)
    Temperature = st.text_input("Temperature:",key=8)
    Hardness = st.text_input("Hardness:",key=111)
    Pain = st.selectbox("Pain:",('Yes','No'),key=101)
    if (Pain == 'Yes'):
        Pain = 1
    else: 
        Pain = 0

    result=""
    if st.button("Predict"):
        result=predict_mastitis(Breed,Months,History,IUFL,EUFL,IUFR,EUFR,IURL,EURL,IURR,EURR,Temperature,Hardness,Pain)
    st.success('The Output is {}' .format(result))
    if st.button("About"):
        st.text("Built by Kawshik")
if __name__=='__main__':
    main()
      