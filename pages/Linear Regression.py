import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn import linear_model
from Home import load_car_data

        
def lr_example():
    
    load_car_data()

    df = st.session_state['data']

    st.title("Linear Regression Example")

    st.write("This is a simple example of linear regression using Streamlit. using streamlit to explore the data can be useful to quickly understand the relationships between variables, and also to find any outliers that may affect the further analysis.")
    
    st.write("This page uses a public dataset of cars to demonstrate a simple linear regression model. The dataset contains various attributes of cars, such as their horsepower, weight, and fuel efficiency.")

    st.write(df.head())

    columns = st.columns(2)

    with columns[0]:
        x_col = st.selectbox("Select X column", df.columns, index=4)
        col0_range = st.slider(f"Select {x_col} range", 
                          min_value=float(df[x_col].min()), 
                          max_value=float(df[x_col].max()), 
                          value=(float(df[x_col].min()), float(df[x_col].max())))
    with columns[1]:
        y_col = st.selectbox("Select Y column", df.columns, index=12)
        col1_range = st.slider(f"Select {y_col} range", 
                          min_value=float(df[y_col].min()), 
                          max_value=float(df[y_col].max()), 
                          value=(float(df[y_col].min()), float(df[y_col].max())))

    lr = linear_model.LinearRegression()
    lr.fit(df[[x_col]], df[y_col])

    fig = px.scatter(
        df.query(f'`{x_col}` >= {col0_range[0]} and `{x_col}` <= {col0_range[1]} and `{y_col}` >= {col1_range[0]} and `{y_col}` <= {col1_range[1]}'), 
        x=x_col, 
        y=y_col, 
        title=f'{x_col} vs {y_col} with Linear Regression',
        trendline='ols', trendline_color_override="red"
    )
    
    st.plotly_chart(fig)
 
    coef, y_intercept = lr.coef_[0], lr.intercept_
    st.write(f"Resulting Equation: y = {coef:.2f} * x + {y_intercept:.2f}")
    st.write(f"This equation tells us that for every unit increase in {x_col}, the {y_col} increases by {coef:.2f} units, with an intercept of {y_intercept:.2f}.")
    st.write("This kind of quick analysis on various dimensions allows quick iteration over ideas, and helps to find the useful features. It is also a good way to find outliers that may affect the model performance. As can be seen in the plot, some of these dimensions have clear outliers that may affect the linear model performance.")


if __name__ == "__main__":
    lr_example()