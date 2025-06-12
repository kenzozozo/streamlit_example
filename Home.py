import streamlit as st
import pandas as pd

def load_car_data():
    if st.session_state.get('data') is None:
        st.session_state['data'] = (
            pd
            .read_csv('https://raw.githubusercontent.com/im-dpaul/EDA-Cars-Data/refs/heads/main/cars_data.csv')
            .query('`highway MPG` < 100 and `city mpg` < 100')
            .dropna(how='any')
        )

def main():
    st.title("Streamlit for Fast Prototyping and Iterative Development")
    st.write("I often use Streamlit to explore datasets and iterate over ML ideas quickly. It is a very handy tool in the process of data analysis and machine learning model development, and showing stakeholders the current status of development in an interactive way.")
    st.write("Note: Some of these pages may take a little while to load, as they are fetching data from the web or performing computations. Please be patient.")
    st.write("You can navigate to the other pages using the sidebar on the left.")
    st.write("I am downloading data from sites on the web, so it may take a little while to load. Here's an example car dataset I downloaded from github.")
    url = "https://raw.githubusercontent.com/im-dpaul/EDA-Cars-Data/refs/heads/main/cars_data.csv"
    st.markdown('[raw.githubusercontent.com/im-dpaul/EDA-Cars-Data](%s)' % url)

    load_car_data()

    st.write(st.session_state['data'].head())

if __name__ == "__main__":
    with st.spinner("Loading data... This might take about 2 minutes..."):
        main()