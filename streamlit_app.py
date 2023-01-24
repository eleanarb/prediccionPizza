from collections import namedtuple
import math
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

"""
# Predicción del Precio de una Pizza con Machine Learning
"""

st.set_page_config(layout="wide")

#Carga de los datos formateados
df = pd.read_csv('pizzaF.csv')
#División del dataset
X = df.drop(['price'], axis=1)
y = df['price']
#Modelacion
model_rf = RandomForestRegressor(random_state=42)
# Entrenamiento
model_rf.fit(X_train, y_train)

# Predicciones
pred_rf = model_rf.predict(X_test)
pred_rf_trn = model_rf.predict(X_train)

prueba = pd.DataFrame({"company":[2], "diameter":[14], "topping":[8], "variant":[1], "size":[5], "extra_sauce":[0], "extra_cheese":[1], "extra_mushrooms":[1]})
precio = (model_rf.predict(prueba))

def form_callback():
    sum = st.session_state.x1 + st.session_state.x2 + st.session_state.x3 + st.session_state.x4
    st.session_state.x1 = st.session_state.x1/sum*100.0
    st.session_state.x2 = st.session_state.x2/sum*100.0
    st.session_state.x3 = st.session_state.x3/sum*100.0
    st.session_state.x4 = st.session_state.x4/sum*100.0
st.metric(label="Precio", value=precio, delta="1.2 °F")

items_company = ("A", "B", "C", "D", "E")
options_company = list(range(len(items_company)))

items_toppings = ("Carne de vaca", "Pimienta Negra", "Pollo", "Carne", "Mozzarella", "Hongos", "Cebollas", "Pepperoni", "Salame", "Carne ahumada", "Atún", "Vegetales")
options_toppings = list(range(len(items_toppings)))

items_variants = ("Carne BBQ Fiesta", "BBQ Salame", "American Classic", "American Favorite", "Classic", "Crunchy", "Double Decker", "Double Mix", "Double Signature", "Extravaganza", "Gourmet greek", "Italian veggie", "Meat Eater", "Meat Lovers", "Neptune tuna", "New york", "Spicy tuna", "Super supreme", "Thai Veggie")
options_variants = list(range(len(items_variants)))

items_size = ("XL", "Jumbo", "Large", "Medium", "Regular", "Small")
options_size = list(range(len(items_size)))

items_extra_sauce = ("Si", "No")
options_extra_sauce = list(range(len(items_extra_sauce)))

items_extra_cheese = ("Si", "No")
options_extra_cheese = list(range(len(items_extra_cheese)))

items_extra_mushrooms = ("Si", "No")
options_extra_mushrooms = list(range(len(items_extra_mushrooms)))

with st.sidebar.form(key='my_form'):
    st.subheader('Glass composition')
    
    company = st.selectbox("Compañia", options_company, format_func=lambda x: items_company[x])

    diameter = st.select_slider(
    'Diámetro de la pizza',
    options=['8', '8.5', '12', '14', '16', '16.5', '17', '18', '18.5', '20', '22' ])
    
    variants = st.selectbox("Pizza", options_variants, format_func=lambda x: items_variants[x])

    toppings = st.selectbox("Topping", options_toppings, format_func=lambda x: items_toppings[x])
    
    size = st.selectbox("Topping", options_size, format_func=lambda x: items_size[x])

    extra_sauce = st.radio("Extra salsa", options_extra_sauce, format_func=lambda x: items_extra_sauce[x])
    
    extra_cheese = st.radio("Extra queso", options_extra_cheese, format_func=lambda x: items_extra_cheese[x])
    
    extra_mushrooms = st.radio("Extra hongos", options_extra_mushrooms, format_func=lambda x: items_extra_mushrooms[x])
    

    submit_button = st.form_submit_button(label='Calculate!', on_click=form_callback)
    composition =  np.array([st.session_state.x1/100, st.session_state.x2/100, st.session_state.x3/100, st.session_state.x4/100]).reshape(1,-1)
