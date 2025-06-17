import streamlit as st
from Home import load_car_data

from sklearn.cluster import ward_tree, dbscan
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from functools import reduce
import numpy as np
import math
import random

import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if 'transform_complete' not in st.session_state:
    st.session_state['transform_complete'] = False

# Initialize PCA min/max values in session state for each category and each component
if 'pca0_min' not in st.session_state:
    st.session_state['pca0_min'] = 0
if 'pca0_max' not in st.session_state:
    st.session_state['pca0_max'] = 0
if 'pca1_max' not in st.session_state:
    st.session_state['pca1_min'] = 0
if 'pca1_max' not in st.session_state:
    st.session_state['pca1_max'] = 0
if 'pca2_min' not in st.session_state:
    st.session_state['pca2_min'] = 0
if 'pca2_max' not in st.session_state:
    st.session_state['pca2_max'] = 0
if 'successful_predictions' not in st.session_state:
    st.session_state['successful_predictions'] = 0
if 'unsuccessful_predictions' not in st.session_state:
    st.session_state['unsuccessful_predictions'] = 0

st.title("Classification Examples")

load_car_data()

st.write("One of my favorite things to do is to turn manual classification tasks into automated ones. Utilizing various methods, I can create automated classification models with high accuracy and explainability.")

st.write("In the cars data, there are several category 'tags' which describe the type or target market for the car. These tags are often compounded, like 'Luxury, Performance' or 'Hybrid, Economy.' The idea is to split these tags apart and then see if it's possible to write a binary classifier (Hybrid: True/False, Luxury: True/False) to classify them based on the other dimensions like MSRP, Horse Power, etc.")

st.write("Here I have written a little classifier experiment to automate classifiaction of the cars data.")

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


st.session_state['unique_categories'] = [] # prep receiver for categories

if st.session_state.get('transform_complete') is False:
    car_data['Market Category'] = car_data['Market Category'].apply(lambda x: x.split(','))
    st.session_state['transform_complete'] = True

if st.session_state['unique_categories'] == []: # this will turn words into letters if run again, so check if exists first
    st.session_state['unique_categories'] = (
        list(reduce(lambda x, y: x.union(y), car_data['Market Category'].apply(set), set()))
    )

unique_fuel_types = car_data['Engine Fuel Type'].unique().tolist()

for category in st.session_state['unique_categories']:
    car_data[category] = car_data['Market Category'].apply(lambda x: 1 if category in x else 0)

for fuel_type in unique_fuel_types:
    car_data[fuel_type] = car_data['Engine Fuel Type'].apply(lambda x: 1 if x == fuel_type else 0)

predictors = predictors + unique_fuel_types

labels = st.session_state['unique_categories']

X = car_data[predictors].values
y = car_data[labels].values

pca = PCA(n_components=3, random_state=42)

reduced_data = pca.fit_transform(X)
# Apply manual axis reduction by shifting the data
st.markdown("## PCA Reduced Data Visualization")
st.write(f"Using PCA, I reduced the number of dimensions of the data from {X.shape[1]} to 3. This allows us to visualize the data in a 3D space. Slicing the data using the selectbox lets us visually inspect if there exists a kind of plane of separation between categories.")

st.write("PCA Components")

three_cols = st.columns(3)

with three_cols[0]:
    st.session_state['pca0_min'], st.session_state['pca0_max'] = (
        st.slider("1st Component", 
        min_value=float(reduced_data[:, 0].min()), 
        max_value=float(reduced_data[:, 0].max()), 
        value=(float(reduced_data[:, 0].min()), float(reduced_data[:, 0].max()))
        )
    )

with three_cols[1]:
    st.session_state['pca1_min'], st.session_state['pca1_max'] = (
        st.slider(
            "3rd Component", 
            min_value=float(reduced_data[:, 1].min()), 
            max_value=float(reduced_data[:, 1].max()), 
            value=(float(reduced_data[:, 1].min()), float(reduced_data[:, 1].max()))
        )
    )
with three_cols[2]:
    st.session_state['pca2_min'], st.session_state['pca2_max'] = (
        st.slider(
            "3rd Component", 
            min_value=float(reduced_data[:, 2].min()), 
            max_value=float(reduced_data[:, 2].max()), 
            value=(float(reduced_data[:, 2].min()), float(reduced_data[:, 2].max()))
        )
    )

selected_category = st.selectbox("Market Category", st.session_state['unique_categories'], index=0, key='market_category')

# Create a mask for the selected category
category_mask = car_data[selected_category] == 1

# Apply the PCA reduction mask to both reduced_data and category_mask
pca_mask = (
    (reduced_data[:, 0] >= st.session_state['pca0_min']) & (reduced_data[:, 0] <= st.session_state['pca0_max']) &
    (reduced_data[:, 1] >= st.session_state['pca1_min']) & (reduced_data[:, 1] <= st.session_state['pca1_max']) &
    (reduced_data[:, 2] >= st.session_state['pca2_min']) & (reduced_data[:, 2] <= st.session_state['pca2_max'])
)

filtered_reduced_data = reduced_data[pca_mask]
filtered_category_mask = category_mask[pca_mask]

# st.write(len(filtered_reduced_data), len(reduced_data), len(filtered_category_mask))

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
st.write("The data shown are projected onto the first three principal components using PCA, reducing the original high-dimensional space to a lower-dimensional representation suitable for visualization. The data below does not represent the original data, it is just the PCA reduced data, so the figures won't make sense in the context of cars.")
st.write(reduced_data[:3])

st.markdown("---")

st.write("# Classification")
st.write("For this experiment, we split data into training and testing sets to evaluate model accuracy. I'll do an 80/20 split.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Length of training set:", len(X_train))
st.write("Length of testing set:", len(X_test))

with st.spinner("Training Decision Tree Classifier..."):

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=59)
    multilabel_random_forest = MultiOutputClassifier(rf_clf)
    multilabel_random_forest.fit(X_train, y_train)

    rf_predictions = multilabel_random_forest.predict(X_test)

    accuracies = {}
    for i, label in enumerate(labels):
        accuracies[label] = accuracy_score(y_test[:, i], rf_predictions[:, i])

    # Turn into a DataFrame
    accuracy_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
    accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)
    accuracy_df['Accuracy (%)'] = (accuracy_df['Accuracy'] * 100).round(2)

    st.write("## Decision Tree Classifier results")

    st.markdown("### Per-Label Accuracy (%)")
    st.write("The accuracy of the model for each label is shown below. In the results I observed, all labels exceeded 95% accuracy. The ones with the lowest accuracy were 'Performance' and 'Hatchback' at 95% and 97% accuracy respectively.")
    st.dataframe(accuracy_df[['Accuracy (%)']])

    cols = 2  # 3 per row
    rows = math.ceil(len(labels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    mcm = multilabel_confusion_matrix(y_test, rf_predictions)

    for i, label in enumerate(labels):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Greens', ax=axes[i],
                    xticklabels=[f'Not {label}', label],
                    yticklabels=[f'Not {label}', label])
        axes[i].set_title(label)

    # Hide extra plots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Practical Example")
    st.write("This may be difficult to imagine how it is useful, so below, we take one random example and show how the predictor labels it against the actual labels. By clicing the button, you can select another single sample and have the model predict its labels.")


    if st.button("Predict 1 result"):
        random_row = car_data.sample(1)
        st.write(random_row[predictors])

        st.write("Market Category for this car: ", random_row['Market Category'].values[0])

        random_predictors = random_row[predictors].values
        random_labels = random_row[labels].values

        #pca it up
        random_reduced = pca.transform(random_predictors)
        # predict based on predictors
        random_prediction = multilabel_random_forest.predict(random_predictors)
        # compare to random_labels
        random_prediction = pd.DataFrame(random_prediction, index=random_row.index)
        random_prediction.columns = labels
        st.write("Existing labels for this car:")
        st.write(pd.DataFrame(random_labels, columns=labels))
        st.write("Predicted Labels for this car:")
        st.write(random_prediction) 
        if (random_prediction == random_labels).all(axis=1).any():
            st.write("Prediction matches actual labels! :)")
            st.session_state['successful_predictions'] += 1
        else:
            st.session_state['unsuccessful_predictions'] += 1
            st.write("Prediction did not match actual labels. :(")

        st.write("Successful Predictions:", st.session_state['successful_predictions'])