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

st.write(
    """
    Bienvenid@ a este dashboard que predice el precio de una pizza a través 
    de algunos parámetros
    """
)
if 'company' not in st.session_state:
    st.session_state.company = 0
    
if 'diameter' not in st.session_state:
    st.session_state.diameter = '8'
    
if 'variants' not in st.session_state:
    st.session_state.variants = 0    
    
if 'toppings' not in st.session_state:
    st.session_state.toppings = 0
    
if 'size' not in st.session_state:
    st.session_state.size = 0    
    
if 'extra_sauce' not in st.session_state:
    st.session_state.extra_sauce = 0 
    
if 'extra_cheese' not in st.session_state:
    st.session_state.extra_cheese = 0 
    
if 'extra_mushrooms' not in st.session_state:
    st.session_state.extra_mushrooms = 0 
    
if 'precio' not in st.session_state:
    st.session_state.precio = 0
    
    
X = pd.read_csv('pizzaX.csv')
y = pd.read_csv('pizzaY.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_rf = RandomForestRegressor(random_state=42)
# Entrenamiento
model_rf.fit(X_train, y_train)


# Predicciones
pred_rf = model_rf.predict(X_test)
pred_rf_trn = model_rf.predict(X_train)

def form_callback():
    companyV =  st.session_state.company
    diameterV = st.session_state.diameter
    variantsV = st.session_state.variants   
    toppingsV = st.session_state.toppings 
    sizeV = st.session_state.size 
    extra_sauceV = st.session_state.extra_sauce
    extra_cheeseV = st.session_state.extra_cheese 
    extra_mushroomsV = st.session_state.extra_mushrooms    
    prueba = pd.DataFrame({"company":[companyV], "diameter":[diameterV], "topping":[toppingsV], "variant":[variantsV], "size":[sizeV], "extra_sauce":[extra_sauceV], "extra_cheese":[extra_cheeseV], "extra_mushrooms":[extra_mushroomsV]})
    precio_ = float((model_rf.predict(prueba)))
    st.session_state.precio = precio_

 

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


col1, col2 = st.columns(2)

with st.form("my_form"):    
    with col1:    
        st.selectbox("Compañia", options_company, format_func=lambda x: items_company[x], key="company") 
        
        st.select_slider(
            'Diámetro de la pizza', 
             options=['8', '8.5', '12', '14', '16', '16.5', '17', '18', '18.5', '20', '22'], key="diameter")
        
        st.selectbox("Pizza", options_variants, format_func=lambda x: items_variants[x], key="variants")
        
        st.selectbox("Topping", options_toppings, format_func=lambda x: items_toppings[x], key="toppings")        
        

    with col2:    
        st.selectbox("Tamaño", options_size, format_func=lambda x: items_size[x], key="size")
        
        st.radio("Extra salsa", options_extra_sauce, format_func=lambda x: items_extra_sauce[x], key="extra_sauce")
        
        st.radio("Extra queso", options_extra_cheese, format_func=lambda x: items_extra_cheese[x], key="extra_cheese")
        
        st.radio("Extra hongos", options_extra_mushrooms, format_func=lambda x: items_extra_mushrooms[x], key="extra_mushrooms")

    
    submit_button = st.form_submit_button(label='Calcular', on_click=form_callback)


st.subheader(f'Precio: $ {st.session_state.precio:,.2f}')

df = pd.read_csv('pizzaO.csv')

option = st.selectbox(
    'Distribución del precio de pizzas',
    ('size', 'variant', 'company', 'topping', 'diameter'))

if option == 'diameter':
    chart = alt.Chart(df).mark_bar().encode(
        x= alt.X("diameter:O", title="Diámetro", axis= alt.Axis(labelAngle=0, labelFontSize=12)),
        y= alt.Y("price:Q", title="Precio", axis = alt.Axis(format = "~s", grid=False)),
    ).properties(
        title = "Distribución del precio de la pizza",
        width = 600
    ).configure_title(
        fontSize=15
    ).configure_view(
        strokeWidth=0
    )   
else:
     chart = alt.Chart(df).mark_bar().encode(
        x= alt.X("diameter:O", title="Diámetro", axis= alt.Axis(labelAngle=0, labelFontSize=12)),
        y= alt.Y("price:Q", title="Precio", axis = alt.Axis(format = "~s", grid=False)),
        color=option
    ).properties(
        title = "Distribución del precio de la pizza",
        width = 600
    ).configure_title(
        fontSize=15
    ).configure_view(
        strokeWidth=0
    )

st.altair_chart(chart, use_container_width=True, theme=None,)
