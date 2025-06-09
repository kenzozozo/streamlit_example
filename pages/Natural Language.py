import streamlit as st
import requests
# tf-idf and nltk
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

def load_data():
    nltk.download('stopwords')

    thucydides = "https://gutenberg.org/cache/epub/7142/pg7142-images.html#link2H_4_0001"
    dickens = "https://www.gutenberg.org/files/98/98-h/98-h.htm"

    if st.session_state.get('thucydides') is None:
        st.session_state['thucydides'] = requests.get(thucydides).text
    if st.session_state.get('dickens') is None:
        st.session_state['dickens'] = requests.get(dickens).text

    thucydides_soup = BeautifulSoup(st.session_state['thucydides'], 'html.parser')
    dickens_soup = BeautifulSoup(st.session_state['dickens'], 'html.parser')
    thucydides_chapters = thucydides_soup.find_all('p')
    dickens_chapters = dickens_soup.find_all('div', class_='chapter')
    thucydides_text = ' '.join([chapter.get_text() for chapter in thucydides_chapters])
    dickens_text = ' '.join([chapter.get_text() for chapter in dickens_chapters])

    # all chapters are in div class="chapter", inside p tags, so get them out using bs4
    return thucydides_text, dickens_text

def main():
    multilabel_exist = False # Hide the buttons before data load

    thucydides, dickens = load_data()
    st.title("Natual Language Processing")
    st.write("I have always been fascinated with processing human languages with computers.")
    
    # show a snippet of either text
    columns = st.columns(2)
    with columns[0]:
        st.subheader("Thucydides: The History of the Peloponnesian War")
        st.write(thucydides[:500])
    with columns[1]:
        st.subheader("Charles Dickens: A Tale of Two Cities")
        st.write(dickens[:500])  
    
    # insert separator
    st.markdown("---")
    
    # Input for x values
    st.write("Let's create a quick LDA analysis of the two texts to see if we can find similar topics between them. This example was taken from the scikit-learn documentation.")
    st.write("https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html")
    
    doc_select = st.selectbox("Select a text to analyze", ["Thucydides", "Dickens"], index=0, key="text_select")

    if st.button("Generate LDA Topics"):
        # Tokenize into documents — use paragraphs (or small chunks)
        thucydides_docs = thucydides.split("\n\n")
        dickens_docs = dickens.split("\n\n")

        # Combine for shared vocabulary
        if doc_select == "Thucydides":
            all_docs = thucydides_docs 
        else:
            all_docs = dickens_docs

        # Vectorize
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(all_docs)

        # LDA
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-11:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(f"Topic {topic_idx}: " + ", ".join(top_features))

        for topic in topics:
            st.write(topic)


if __name__ == "__main__":
    main()