import streamlit as st
from Home import load_car_data
from sklearn.cluster import ward_tree, dbscan
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.tree import DecisionTreeClassifier

if 'transform_complete' not in st.session_state:
    st.session_state['transform_complete'] = False

st.title("Classification Examples")

load_car_data()

st.write("One of my favorite things to do is to turn manual classification tasks into automated ones. Utilizing various methods, I can create automated classification models with high accuracy and explainability.")

from sklearn.decomposition import PCA

st.write("Perform PCA on existing data to extract most meaningful dimensions.")

pca = PCA()

car_data = st.session_state['data']

numbers = [
    "MSRP",
    "Popularity",
    "highway MPG", 
    "Number of Doors",
    "Engine Cylinders",
    "Engine HP"
    ]

targets = ["Market Category"]

classes = [
    "Make",
    "Model",
    "Engine Fuel Type", 
    "Transmission Type", 
    "Driven_Wheels",
    "Market Category",
    "Vehicle Size",
    "Vehicle Style"
    ]

from functools import reduce

unique_categories = []

st.write(st.session_state.get('transform_complete'))

if st.session_state.get('transform_complete') is False:
    car_data['Market Category'] = car_data['Market Category'].apply(lambda x: x.split(','))
    st.session_state['transform_complete'] = True

if unique_categories == []:
    unique_categories = reduce(lambda x, y: x.union(y), car_data['Market Category'].apply(set), set())

unique_fuel_types = car_data['Engine Fuel Type'].unique().tolist()

for category in unique_categories:
    car_data[category] = car_data['Market Category'].apply(lambda x: 1 if category in x else 0)

for fuel_type in unique_fuel_types:
    car_data[fuel_type] = car_data['Engine Fuel Type'].apply(lambda x: 1 if x == fuel_type else 0)

predictors = [
    "MSRP",
    "Popularity",
    "highway MPG", 
    "Number of Doors",
    "Engine Cylinders",
    "Engine HP"
    ] + unique_fuel_types

targets = unique_categories

st.write(unique_categories)
st.write(unique_fuel_types)
st.write(car_data)