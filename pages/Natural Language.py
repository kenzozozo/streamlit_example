import streamlit as st
import requests
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import textwrap

import numpy as np
import nltk

if 'punkt_tab' not in nltk.data.path:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('rslp')

def load_data():

    thucydides = "https://gutenberg.org/cache/epub/7142/pg7142.txt"
    doyle = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
    dickens = "https://gutenberg.org/cache/epub/98/pg98.txt"
    melville = "https://gutenberg.org/cache/epub/2701/pg2701.txt"

    def clean_text(text):
        stemmer = nltk.stem.RSLPStemmer()
        stopwords = nltk.corpus.stopwords.words('english')
        tmp = text
        tmp = tmp.lower()  # Convert to lowercase
        tmp = stemmer.stem(tmp)  # Stem the text
        tmp = ' '.join([word for word in text.split() if word not in stopwords])
        return tmp

    if st.session_state.get('thucydides') is None:
        thu_text = requests.get(thucydides).text
        st.session_state['thucydides'] = clean_text(thu_text)
    if st.session_state.get('dickens') is None:
        dickens_text = requests.get(dickens).text
        st.session_state['dickens'] = clean_text(dickens_text)
    if st.session_state.get('melville') is None:
        melville_text = requests.get(melville).text
        st.session_state['melville'] = clean_text(melville_text)
    if st.session_state.get('doyle') is None:
        doyle_text = requests.get(doyle).text
        st.session_state['doyle'] = clean_text(doyle_text)

def main():
    topics = False # prepping vars so no error on initial load
    probs = False
    multilabel_exist = False # Hide the buttons before data load

    load_data()

    # Chunking function to accommodate bertopic < 5 docs long
    def chunk_text(text, chunk_size=500):
        return [
            chunk.strip().replace("\n", " ") 
            for chunk in textwrap.wrap(text, width=chunk_size) 
            if len(chunk.strip()) > 50
        ]

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

            # Encode embeddings
            with st.spinnter("Encoding embeddings... This might take a while..."):
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)

            # Fit BERTopic
            topic_model = BERTopic(umap_model=None)
            with st.spinner("Fitting BERTopic model... This might take a while..."):
                topics, probs = topic_model.fit_transform(all_chunks, embeddings)

            # Store everything
            st.session_state["bertopic_model"] = topic_model
            st.session_state["bertopic_chunks"] = all_chunks
            st.session_state["bertopic_sources"] = chunk_sources
            st.session_state["bertopic_topics"] = topics
            st.session_state["bertopic_probs"] = probs


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

    # Streamlit dropdown
    doc_select = st.selectbox("Select a book", ["Select One"] + titles)

    # Button and logic
    if st.button("Generate Topics"):
        with st.spinner("Generating topics..."):
            if doc_select == "Select One":
                st.error("Please select a valid text.")
                        
            else:
                st.session_state['bertopic_model'] = st.session_state.get('bertopic_model', None)
                if st.session_state['bertopic_model'] is None:
                    st.error("BERTopic model is not ready. Please wait for it to load.")
                else:
                    # Get the selected text
                    selected_text = doc_dict[doc_select]
                    
                    # Chunk the text
                    chunks = chunk_text(selected_text)
                    
                    # Encode embeddings
                    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
                    
                    # Transform with BERTopic
                    topics, probs = st.session_state['bertopic_model'].transform(chunks, embeddings)
                    
                    # Display results
                    st.subheader(f"Topics for {doc_select}")
                    st.write(f"Found {len(set(topics))} topics in the selected text.")
                    
                    # Show topics and probabilities
                    for topic in set(topics):
                        if topic != -1:
                            topic_words = st.session_state['bertopic_model'].get_topic(topic)
                            topic_prob = np.mean(probs[topics == topic])
                            st.write(f"**Topic {topic}:** {', '.join([word for word, _ in topic_words])} (Probability: {topic_prob:.2f})")                

                    # # Show topic distribution
                    # topic_counts = np.bincount(topics[topics != -1])

if __name__ == "__main__":
    with st.spinner("Loading data and creating bertopic model... This might take a little while..."):
        load_data()
        main()