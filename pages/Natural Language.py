import streamlit as st
import requests
# tf-idf and nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from bs4 import BeautifulSoup

def load_data():
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
    st.title("Thucydides: The History of the Peloponnesian War")
    st.write(thucydides[:1000])
    st.title("Charles Dickens: A Tale of Two Cities")
    st.write(dickens[:1000])  
    
    # insert separator
    st.markdown("---")
    
    # Input for x values
    st.write("Let's create a quick LDA analysis of the two texts to see if we can find similar topics between them. This example was taken from the scikit-learn documentation.")
    st.write("https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html")
    if st.button("Generate Multilabel Classification Data"):
        X, y = make_multilabel_classification(random_state=85)
        multilabel_exist = True

        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        lda.fit_transform(X)
        st.session_state['lda_model'] = lda
        # Display the topics


    if 'lda_model' in st.session_state:
        lda = st.session_state['lda_model']
        st.write("LDA Model Topics:")
        for idx, topic in enumerate(lda.components_):
            feature_names = [chr(i + 97) for i in range(len(topic))]
            # st.write(f"Topic {idx + 1}:")
            # st.write([f"Word {i}: {word}" for i, word in enumerate(topic.argsort()[-10:])])   
            st.write(feature_names)     

if __name__ == "__main__":
    main()