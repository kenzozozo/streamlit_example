import streamlit as st
from Home import load_car_data
from sklearn.cluster import ward_tree, dbscan
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

if 'transform_complete' not in st.session_state:
    st.session_state['transform_complete'] = False

st.title("Classification Examples")

load_car_data()

st.write("One of my favorite things to do is to turn manual classification tasks into automated ones. Utilizing various methods, I can create automated classification models with high accuracy and explainability.")

st.write("Here I have written a little classifier to automate the classifiaction of the cars data.")

st.markdown("---")

car_data = st.session_state['data']

predictors = [
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
import numpy as np

unique_categories = []

if st.session_state.get('transform_complete') is False:
    car_data['Market Category'] = car_data['Market Category'].apply(lambda x: x.split(','))
    st.session_state['transform_complete'] = True

if unique_categories == []:
    unique_categories = list(reduce(lambda x, y: x.union(y), car_data['Market Category'].apply(set), set()))

unique_fuel_types = car_data['Engine Fuel Type'].unique().tolist()

for category in unique_categories:
    car_data[category] = car_data['Market Category'].apply(lambda x: 1 if category in x else 0)

for fuel_type in unique_fuel_types:
    car_data[fuel_type] = car_data['Engine Fuel Type'].apply(lambda x: 1 if x == fuel_type else 0)

predictors = predictors + unique_fuel_types

labels = unique_categories

X = car_data[predictors].values
y = car_data[labels].values

pca = PCA(n_components=3, random_state=42)

reduced_data = pca.fit_transform(X)

# Apply manual axis reduction by shifting the data
st.markdown("## PCA Reduced Data Visualization")
st.write(f"Using PCA, I reduced the number of dimensions of the data from {X.shape[1]} to 3. This allows us to visualize the data in a 3D space.")
selected_category = st.selectbox("Market Category", unique_categories, index=0, key='market_category')

st.write("PCA Components")

three_cols = st.columns(3)

with three_cols[0]:
    pca1_min, pca1_max = st.slider("1st Component", 
        min_value=float(reduced_data[:, 2].min()), 
        max_value=float(reduced_data[:, 2].max()), 
        value=(float(reduced_data[:, 2].min()), float(reduced_data[:, 2].max()))
        )
with three_cols[1]:
    pca2_min, pca2_max = st.slider("2nd Component", 
        min_value=float(reduced_data[:, 2].min()), 
        max_value=float(reduced_data[:, 2].max()), 
        value=(float(reduced_data[:, 2].min()), float(reduced_data[:, 2].max()))
        )
with three_cols[2]:
    pca3_min, pca3_max = st.slider("3rd Component", 
        min_value=float(reduced_data[:, 2].min()), 
        max_value=float(reduced_data[:, 2].max()), 
        value=(float(reduced_data[:, 2].min()), float(reduced_data[:, 2].max()))
        )
    # Create a mask for the selected category
    category_mask = car_data[selected_category] == 1

    # Apply the PCA reduction mask to both reduced_data and category_mask
    pca_mask = (
        (reduced_data[:, 0] >= pca1_min) & (reduced_data[:, 0] <= pca1_max) &
        (reduced_data[:, 1] >= pca2_min) & (reduced_data[:, 1] <= pca2_max) &
        (reduced_data[:, 2] >= pca3_min) & (reduced_data[:, 2] <= pca3_max)
    )

    filtered_reduced_data = reduced_data[pca_mask]
    filtered_category_mask = category_mask[pca_mask]

    # Assign colors: red for selected category, blue for others
    colors = np.where(filtered_category_mask, 'red', 'gray')

    fig = px.scatter_3d(
        x=filtered_reduced_data[:, 0],
        y=filtered_reduced_data[:, 1],
        z=filtered_reduced_data[:, 2],
        color=colors,
        color_discrete_map={'red': 'red' , 'gray': 'gray'},
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'z': 'PCA Component 3'},
        title=f"3D PCA Scatter Plot: {selected_category} in Red",
        size_max=1,
        height=800,
    )

st.plotly_chart(fig)
st.write("The data here represent the dimensions projected onto the first three principal components using PCA, it is projecting the high dimensional data into a lower dimension.")
st.write(reduced_data[:4])

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X, y.argmax(axis=1))
rf_predictions = rf_clf.predict(X)

# confusion matrix alignment
cm_rf = confusion_matrix(y.argmax(axis=1), rf_predictions)
fig_rf, ax_rf = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax_rf,
            xticklabels=labels, yticklabels=labels)
st.markdown("### Random Forest Classifier Confusion Matrix")
st.pyplot(fig_rf)

st.write(X)