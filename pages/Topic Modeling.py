import streamlit as st
import requests
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import textwrap

import numpy as np
import nltk

# Ensure required NLTK resources are available, download if missing
required_resources = ['punkt', 'stopwords']
for resource in required_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

def load_data():

    thucydides = "https://gutenberg.org/cache/epub/7142/pg7142.txt"
    doyle = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
    dickens = "https://gutenberg.org/cache/epub/98/pg98.txt"
    melville = "https://gutenberg.org/cache/epub/2701/pg2701.txt"

    def clean_text(text):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        # Lowercase and split into words
        words = text.lower().split()
        return ' '.join(words)

    if st.session_state.get('thucydides') is None:
        try:
            thu_text = requests.get(thucydides).text
        except:
            with open("data/thucydides.txt", "r", encoding='utf-8') as file:
                thu_text = file.read()
        st.session_state['thucydides'] = clean_text(thu_text)
    if st.session_state.get('dickens') is None:
        try:
            dickens_text = requests.get(dickens).text
        except:
            with open("data/dickens.txt", "r", encoding='utf-8') as file:
                dickens_text = file.read()
        st.session_state['dickens'] = clean_text(dickens_text)
    if st.session_state.get('melville') is None:
        try:
            melville_text = requests.get(melville).text
        except:
            with open("data/melville.txt", "r", encoding='utf-8') as file:
                melville_text = file.read()
        st.session_state['melville'] = clean_text(melville_text)
    if st.session_state.get('doyle') is None:
        try:
            doyle_text = requests.get(doyle).text
        except:
            with open("data/doyle.txt", "r", encoding='utf-8') as file:
                doyle_text = file.read()
        st.session_state['doyle'] = clean_text(doyle_text)
    if 'embedding_model' not in st.session_state:
        # st.session_state['embedding_model'] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        st.session_state['embedding_model'] = SentenceTransformer("sentence-transformers/gooaq")

def main():
    topics = False # prepping vars so no error on initial load
    probs = False
    multilabel_exist = False # Hide the buttons before data load

    load_data()

    # Chunking function to accommodate bertopic < 5 docs long
    def chunk_text(text, chunk_size=5):
        sentences = nltk.sent_tokenize(text)
        chunks = [
            ' '.join(sentences[i:i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
        return [chunk for chunk in chunks if len(chunk.split()) > 20]

    doc_dict = {
        "Thucydides": st.session_state['thucydides'],
        "Dickens": st.session_state['dickens'],
        "Melville": st.session_state['melville'],
        "Doyle": st.session_state['doyle']
    }

    # Chunk and store if not already cached
    if "bertopic_model" not in st.session_state:
        with st.spinner("Preparing BERTopic model..."):
            # Chunk all books and track source

            all_chunks = []
            chunk_sources = []
            for name, book in doc_dict.items():
                with st.spinner(f"Processing {name}..."):
                    chunks = chunk_text(book)
                    all_chunks.extend(chunks)
                    chunk_sources.extend([name] * len(chunks))

            # Encode text and create embeddings
            with st.spinner("Encoding text... Creating embeddings... This might take a while..."):
                embeddings = st.session_state['embedding_model'].encode(all_chunks)

            # Fit BERTopic
            topic_model = BERTopic(umap_model=None, nr_topics=5, calculate_probabilities=True, verbose=True)
            with st.spinner("Fitting & transforming with BERTopic model... This might take a while..."):
                topics, probs = topic_model.fit_transform(all_chunks, embeddings)

            # Store everything
            st.session_state["bertopic_model"] = topic_model
            st.session_state["bertopic_chunks"] = all_chunks
            st.session_state["bertopic_sources"] = chunk_sources
            st.session_state["bertopic_topics"] = topics
            st.session_state["bertopic_probs"] = probs

    # DEBUG #
    st.write(f"Debug BERTopic model loaded with {len(st.session_state['bertopic_chunks'])} chunks {len(set(st.session_state['bertopic_topics']))} topics.")
    st.write(f"Debug {len(st.session_state['bertopic_sources'])} sources: {set(st.session_state['bertopic_sources'])}")
    st.write(f"Debug {len(st.session_state['bertopic_topics'])} topics: {set(st.session_state['bertopic_topics'])}")

    # DEBUG #

    st.title("Natual Language Processing")
    st.write("I have always been fascinated with processing human languages with computers. My newest favorite tool is topic modeling using Latent Dirichlet Allocation (LDA). This page is a simple example of how to use LDA to find topics in text data. It uses the scikit-learn library to perform the LDA analysis.")
    st.write("Here are a few texts that I downloaded for analysis from project Gutenberg...")
    st.write("(If it takes a while to load, please be patient. The texts are quite large and the LDA analysis can take some time.)")
    st.markdown("---")

    columns = st.columns(2)
    with columns[0]:
        st.subheader("Thucydides: The History of the Peloponnesian War")
        st.write(st.session_state['thucydides'][5000:5500])
    with columns[1]:
        st.subheader("Charles Dickens: A Tale of Two Cities")
        st.write(st.session_state['dickens'][5000:5500])  

    second_columns = st.columns(2)
    with second_columns[0]:
        st.subheader("Herman Melville: Moby Dick")
        if st.session_state['melville']:
            st.write(st.session_state['melville'][10000:10500])
    with second_columns[1]:
        st.subheader("Arthur Conan Doyle: The Adventures of Sherlock Holmes")
        if st.session_state['doyle']:
            st.write(st.session_state['doyle'][5000:5500])
    
    # insert separator
    st.markdown("---")
    
    # Input for x values
    st.write("Below we can see the topics extracted by the BERTopic model. With this, we could then ask someone to manually classify this grouping of topics to help make it more human readable.")
    st.write("https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html")

    titles = list(doc_dict.keys())

    # Button and logic
    if st.button("Generate Topics"):
        with st.spinner("Generating topics..."):
            # show all topics
            st.write(st.session_state['bertopic_model'].get_topic_info())
                
if __name__ == "__main__":
    with st.spinner("Loading data and creating bertopic model... This might take a little while..."):
        load_data()
        main()