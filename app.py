import streamlit as st
from transformers import pipeline
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the summarizer for abstractive summarization
summarizer = pipeline("summarization")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the extractive summarization function
def textrank_summary(text, top_n=5):
    sentences = sent_tokenize(text)
    tfidf = TfidfVectorizer(stop_words=stopwords.words('english')).fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf, tfidf)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [sentence for _, sentence in ranked_sentences[:top_n]]
    summary = ' '.join(summary_sentences)
    return summary

# Define the Streamlit app
def main():
    st.title("Text Summarization App")

    st.sidebar.title("Summarization Options")
    option = st.sidebar.selectbox("Choose Summarization Type", ("Extractive", "Abstractive"))

    text = st.text_area("Enter the text you want to summarize", height=300)
    
    if st.button("Summarize"):
        if option == "Extractive":
            summary = textrank_summary(text)
        else:
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        
        st.write("### Summary")
        st.write(summary)

if __name__ == "__main__":
    main()
