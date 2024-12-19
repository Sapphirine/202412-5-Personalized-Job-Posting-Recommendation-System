from venv import logger
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import PyPDF2
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import re
import torch
import nltk
from nltk.corpus import stopwords
import os
import sys

module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from resume_parser import resume_parser
from resume_info_extraction import resume_summary
from data_cleaning import clean_data

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

class JobRecommender:
    def __init__(self):
        self.job_embedding_bert = torch.load('./data/job_embeddings_bert.pt')
        self.job_embedding_tfidf = pickle.load(open('./data/tfidf_job_descriptions.pkl', 'rb'))
        with open('./data/tfidf_model.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
    def compute_resume_embedding(self, resume_sum, resume_info):
        model = SentenceTransformer('all-mpnet-base-v2')
        employment_exp = resume_info['employment_experiences']

        self.resume_embedding_bert = model.encode(resume_sum, convert_to_tensor=True)
        self.resume_tfidf = self.tfidf_vectorizer.transform([employment_exp])

        return self.resume_embedding_bert, self.resume_tfidf
    
    

    
    def generate_recommendations(self, job_embeddings_bert, tfidf_job_descriptions, resume_embedding_bert, resume_tfidf, df, top_k=10):
        
        
        cosine_scores_bert = util.cos_sim(resume_embedding_bert, job_embeddings_bert)[0].cpu().numpy()
        cosine_scores_tfidf = cosine_similarity(resume_tfidf, tfidf_job_descriptions).flatten()
        combined_scores = 0.5 * cosine_scores_bert + 0.5 * cosine_scores_tfidf
        
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        recommendations = []
        
        for idx in top_indices:
            job = df.iloc[idx]
            recommendations.append({
                'title': job['title'],
                'company': job['company'],
                'location': job['city'],
                'similarity_score': combined_scores[idx],
                'description': job['description'], 
                'job_url': job['job_url'], 
                'days_old': job['days_old']
            })


        return recommendations
    
    
def process_resume(uploaded_file):

    try:
        if uploaded_file.type == "application/pdf":
            resume_sum = resume_summary(uploaded_file)
            resume_info = resume_parser(uploaded_file)
            return resume_sum, resume_info
        else:
            raise ValueError("Unsupported file type. Please upload a PDF file.")
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise
    

def main():
    st.set_page_config(page_title="Job Recommendation System", layout="wide")
    
    # Title and description
    st.title("🎯 Job Recommendation System 🎯 ")
    st.write("Upload your resume and get personalized job recommendations!")
    
    # Sidebar for inputs
    with st.sidebar:
        
        st.header("Upload Resume")
        uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
        
        st.header("Job Preferences")
        num_recommendations = st.slider(
                "Number of Recommendations",
                min_value=1,
                max_value=25,  # You can adjust this maximum value
                value=10,  # Default value
                step=1,
                help="Select how many job recommendations you want to see"
            )
        
    if uploaded_file is not None:
        with st.spinner("Processing your resume..."):
            resume_sum, resume_info = process_resume(uploaded_file)
                
            
            # Initialize recommender and generate recommendations
            recommender = JobRecommender()
            
            
            df = pd.read_csv('./data/job_data.csv') 
            df = clean_data(df)
            
            resume_embedding_bert, resume_tfidf = recommender.compute_resume_embedding(resume_sum, resume_info)
            recommendations = recommender.generate_recommendations(
                recommender.job_embedding_bert,
                recommender.job_embedding_tfidf,
                resume_embedding_bert,
                resume_tfidf,
                df,
                num_recommendations
            )
            
            st.session_state.recommendations = recommendations
                
    # Display recommendations
    if st.session_state.recommendations:
        st.header("Top Job Recommendations")
        
        for i, job in enumerate(st.session_state.recommendations, 1):
            with st.expander(f"**{i}: {job['title']}** at **{job['company']}**"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**🏢 Company Details**")
                    st.write(f"**Company:** {job['company']}")
                    st.write(f"**Location:** {job['location']}")
                    if job['job_url'] != '#':
                        st.write("**Job URL:**", job['job_url'])
                
                with col2:
                    st.markdown("**📅 Age**")
                    st.write(f"**Posted:** {job['days_old']} days ago")
                    st.markdown("**🎯 Match Details**")
                    st.write(f"**Match Score:** {job['similarity_score']*1.5 :.2%}")
                    
                
                # Job description takes full width
                st.markdown("---")
                st.markdown("**📝 Job Description**")
                st.write(job['description'])
            
            # Add some spacing between job listings
            st.write("")
    



if __name__ == "__main__":
    main()

