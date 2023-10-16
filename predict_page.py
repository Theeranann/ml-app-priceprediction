import streamlit as st
import pickle
import numpy as np
import locale

# Set the locale to the user's default
locale.setlocale(locale.LC_ALL, '')


def load_model():
    with open('motorcycle_saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_Motorcycle = data["le_Motorcycle"]
le_Color = data["le_Color"]
le_Year = data["le_Year"]

def show_predict_page():
    st.title("ประเมินราคารถจักรยานยนต์มือสอง")

    st.write("""### เราต้องการข้อมูลบางอย่างเพื่อทำนายราคา""")

    motorcycles = (
        'CLICK125I', 'ZOOMER-X', 'PCX150', 'SCOOPY-I', 'FORZA300',
       'N-MAX155', 'AEROX155', 'Q-BIX125', 'N-MAX', 'FINO125I',
       'GRAND FILANO', 'CLICK110I', 'WAVE125I', 'FINN115I', 'Q-BIX',
       'GT125', 'PCX160', 'CLICK150I', 'FILANO', 'CLICK125', 'MIO125',
       'FINO125', 'MOOVE', 'FINO', 'PCX125', 'MIO125I', 'FINO FI',
       'NOUVO MX', 'CLICK', 'QBIX', 'SPACY-I', 'FINO PS', 'MIO',
       'WAVE110I', 'AIRBLADE', 'CLICK-I', 'DREAM SUPER CUB', 'EXCITER',
       'WAVE125R', 'SONIC125', 'WAVE100', 'WAVE125S', 'SONIC 125',
    )

    colors = (
        'white', 'black', 'red', 'green', 'brown', 'pink', 'blue', 'gray',
       'orange', 'yellow', 'Other', 'purple',
    )

    years = (
    2565, 2564, 2563, 2562, 2561, 2560, 2559, 2558, 2557, 
    2556, 2555, 2554, 2553, 2552, 2551, 2550, 2549, 2548, 
    2547, 2546, 2545, 2544, 2540
    )

    motorcycle = st.selectbox("motorcycle", motorcycles)
    color = st.selectbox("color", colors)
    Year = st.selectbox("Year", years)
    
    mileage = st.slider("Mileage", 0, 9000, 100)

    ok = st.button("Calculate")
    if ok:
        X = np.array([[motorcycle, color, Year, mileage ]])
        X[:, 0] = le_Motorcycle.transform(X[:,0])
        X[:, 1] = le_Color.transform(X[:,1])
        X[:, 2] = le_Year.transform(X[:,2])
        X = X.astype(float)

        price = regressor.predict(X)
        formatted_price = locale.format_string("%.2f", price[0], grouping=True)
        st.subheader(f"ราคาโดยประมาณอยู่ที่ {formatted_price} บาท")
        # st.subheader(f"The estimated price is ${price[0]:.2f}")
