import pandas as pd
import numpy as np
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import AutoModel
import torch

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import pickle

import warnings
warnings.filterwarnings('ignore')


def compute_job_embeddings(df):
    model = SentenceTransformer('all-mpnet-base-v2')
    job_embeddings_bert = model.encode(df['cleaned_desc'].tolist(), convert_to_tensor=True)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_job_descriptions = tfidf_vectorizer.fit_transform(df['description_clean'])

    with open('./data/tfidf_model.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    with open('./data/tfidf_job_descriptions.pkl', 'wb') as f:
        pickle.dump(tfidf_job_descriptions, f)

    torch.save(job_embeddings_bert, './data/job_embeddings_bert.pt')