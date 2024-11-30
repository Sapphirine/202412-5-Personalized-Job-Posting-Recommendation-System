import pandas as pd
import numpy as np
import re 


def extract_filtered_bullet_texts(job_description):

    # Define keywords to filter out
    excluded_keywords = ['equal', 'disability', 'veteran', 'criminal', 'e\-verify', '401(k)', 'insurance', '----']

    # Regex to match bullet points and the text following them
    bullet_texts = re.findall(r'^\s*[\*\-\•\d+\.]\s*(.+)', job_description, re.MULTILINE)

    # Filter out bullet points containing excluded keywords (case-insensitive)
    filtered_bullets = [
        text.strip() for text in bullet_texts
        if not any(keyword in text.lower() for keyword in excluded_keywords)
    ]
    # if a sentence starts with or end with *, delete the sentence        
    filtered_bullets = [x for x in filtered_bullets if not x.startswith('*') and not x.endswith('*')]
    # if a sentence only contains numbers, delete the sentence
    filtered_bullets = [x for x in filtered_bullets if not x.isdigit()]
    # if a sentence only contains 2 or fewer words, delete the sentence
    filtered_bullets = [x for x in filtered_bullets if len(x.split()) > 2]

    return filtered_bullets


def clean_data(df):
    df = df[['id', 'site', 'job_url', 'title', 'company', 'location', 'date_posted', 'description', 'date_fetched']]
    df = df.dropna(subset=['description', 'title', 'company'])
    df['location'] = df['location'].fillna('')  # Fill NaN values with empty string
    df['location_split'] = df['location'].str.split(',').apply(lambda x: len(x) if x else 0)

    df = df[df['location_split'] > 1]
    df['city'] = df['location'].str.split(',').apply(lambda x: x[0].strip())
    df['state'] = df['location'].str.split(',').apply(lambda x: x[1].strip())
    df.drop(columns=['location', 'location_split'], inplace=True)

    df = df[~(df['state'] == 'US') & (df['city'] != 'Remote')]
    df['cleaned_desc'] = df['description'].apply(extract_filtered_bullet_texts)
    df['cleaned_desc_len'] = df['cleaned_desc'].apply(len)
    df = df[(df['cleaned_desc_len'] > 4) & (df['cleaned_desc_len'] < 40)]

    df['cleaned_desc'] = df['cleaned_desc'].apply(lambda x: ' '.join(x))
    
    return df