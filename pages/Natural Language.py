import streamlit as st
import requests
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# download punkt-tab if not already

if 'punkt_tab' not in nltk.data.path:
    nltk.download('punkt')
    nltk.download('punkt_tab', download_dir=nltk.data.path[0])

def load_data():

    thucydides = "https://gutenberg.org/cache/epub/7142/pg7142.txt"
    doyle = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
    dickens = "https://gutenberg.org/cache/epub/98/pg98.txt"
    melville = "https://gutenberg.org/cache/epub/2701/pg2701.txt"

    if st.session_state.get('thucydides') is None:
        st.session_state['thucydides'] = requests.get(thucydides).text
    if st.session_state.get('dickens') is None:
        st.session_state['dickens'] = requests.get(dickens).text
    if st.session_state.get('melville') is None:
        st.session_state['melville'] = requests.get(melville).text
    if st.session_state.get('doyle') is None:
        st.session_state['doyle'] = requests.get(doyle).text

    thucydides_text = st.session_state.get('thucydides', None)
    dickens_text = st.session_state.get('dickens', None)
    melville_text = st.session_state.get('melville', None)
    doyle_text = st.session_state.get('doyle', None)

    # all chapters are in div class="chapter", inside p tags, so get them out using bs4
    return thucydides_text, dickens_text, melville_text, doyle_text

def main():
    multilabel_exist = False # Hide the buttons before data load

    st.title("Natual Language Processing")
    st.write("I have always been fascinated with processing human languages with computers. My newest favorite tool is topic modeling using Latent Dirichlet Allocation (LDA). This page is a simple example of how to use LDA to find topics in text data. It uses the scikit-learn library to perform the LDA analysis.")
    st.write("Here are a few texts that I downloaded for analysis from project Gutenberg...")
    st.write("(If it takes a while to load, please be patient. The texts are quite large and the LDA analysis can take some time.)")
    st.markdown("---")

    thucydides, dickens, melville, doyle = load_data()

    # show a snippet of either text
    columns = st.columns(2)
    with columns[0]:
        st.subheader("Thucydides: The History of the Peloponnesian War")
        st.write(thucydides[5000:5500])
    with columns[1]:
        st.subheader("Charles Dickens: A Tale of Two Cities")
        st.write(dickens[5000:5500])  

    second_columns = st.columns(2)
    with second_columns[0]:
        st.subheader("Herman Melville: Moby Dick")
        if melville:
            st.write(melville[10000:10500])
    with second_columns[1]:
        st.subheader("Arthur Conan Doyle: The Adventures of Sherlock Holmes")
        if doyle:
            st.write(doyle[5000:5500])
    
    # insert separator
    st.markdown("---")
    
    # Input for x values
    st.write("Let's create a quick LDA analysis of the two texts to see if we can find similar topics between them. This example was taken from the scikit-learn documentation.")
    st.write("https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html")
    
    doc_select = st.selectbox(
        "Select a text to analyze", 
        ["Select One", "Thucydides", "Dickens", "Melville", "Doyle"], 
        index=0, 
        key="text_select"
    )

    doc_dict = {
        "Thucydides": thucydides,
        "Dickens": dickens,
        "Melville": melville,
        "Doyle": doyle
    }

    if st.button("Generate LDA Topics"):
        # show loading 
        with st.spinner("Generating topics... This may take a while..."):
            # Check if the selected text is valid
            if doc_select not in doc_dict:
                st.error("Please select a valid text to analyze.")
                return

        # Combine for shared vocabulary
        if doc_select == "Select One":
            st.error("Please select a text to analyze.")
            return
        else:
            doc = [doc_dict[doc_select]]
            topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(doc)

    st.write(topics, probs)

if __name__ == "__main__":
    main()