import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 

def extract_info(text):
    """
    Extracts the first paragraph and bullet points following each title in the text.

    Parameters:
    - text (str): The input text from which to extract information.

    Returns:
    - List[Dict]: A list of dictionaries containing the title, first paragraph, and bullet points.
    """
    # Pattern to find titles in the format **title**
    pattern_title = r'\*\*(.*?)\*\*'

    excluded_keywords = ['equal', 'disability', 'veteran', 'criminal', 'e\-verify', '401(k)', 'insurance', '----', 'tuition', 'vacation', 
                         'holiday', 'coverage', 'community service', 'paid parental leave', 'discount', 'family', 'employee']
    title_excluded_keywords = ['company', 'benefits', 'benefit', 'who we are', 
                               'diversity', 'belonging', 'compensation', 'diversity', 
                               'equal', 'do', 'office', 'offices', 'why', 'interview', 'opportunity', 'about', 'perks', 'fits and pe', 
                               'under', 'designation', 'hybrid', 'remote', 'salary', 'date', 'who', 'visa', 'commitment', 'electrical', 'id', 'travel']

    # Find all titles with their positions
    titles = [(m.group(1).strip(), m.start(), m.end()) for m in re.finditer(pattern_title, text)]

    # Add a sentinel at the end to handle the last title

    titles.append(('', len(text), len(text)))
    
    # if title contains title_excluded_keywords, drop 
    titles = [title for title in titles if not any(keyword in title[0].lower() for keyword in title_excluded_keywords)]
    

    extracted_data = ''

    # Iterate over each title
    for i in range(len(titles) - 1):
        title_text = titles[i][0]
        start_content = titles[i][2]  # End position of the current title
        end_content = titles[i + 1][1]  # Start position of the next title

        # Extract content between the current title and the next title
        content = text[start_content:end_content]

        # Extract the first paragraph after the title
        paragraphs = re.split(r'\n\s*\n', content.strip(), maxsplit=1)
        first_paragraph = paragraphs[0].strip() if paragraphs else ''

        # Extract bullet points that start with '*'
        bullet_points = re.findall(r'^\s*\*\s*(.*)', content, re.MULTILINE)

        excluded_keywords = ['equal', 'disability', 'veteran', 'criminal', 'e\-verify', '401(k)', 'insurance', '----', 'tuition', 'vacation', 'holiday', 'coverage', 'community service', 'paid parental leave', 'discount', 'family', 'employee']

        # Filter out sentences that contain excluded keywords
        bullet_points = [point.strip() for point in bullet_points if not any(keyword in point.lower() for keyword in excluded_keywords)]

        # delete the symbol in front of bullet points 
        # Append the extracted information to the list
        extracted_data += f"{first_paragraph}\n"
        extracted_data += f"{''.join(bullet_points)}\n\n"

    # remove any blank spaces 
    extracted_data = ' '.join(extracted_data.split())
    extracted_data = extracted_data.replace('  ', ' ')
    extracted_data = extracted_data.replace('*', ' ')
    extracted_data = extracted_data.replace('-', ' ')
    
    # limit the length of the extracted data to 2500 words 
    extracted_data = extracted_data[:2500]
    # find the last sentence in the extracted data
    last_sentence = extracted_data.rfind('.')
    extracted_data = extracted_data[:last_sentence+1]
    

    return extracted_data

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def clean_data(df):
    print(df.shape)
    df = df[['id', 'site', 'job_url', 'title', 'company', 'location', 'date_posted', 'description', 'date_fetched']]
    df = df.dropna(subset=['description', 'title', 'company'])
    df['title_lower'] = df['title'].str.lower()
    df['company_lower'] = df['company'].str.lower()
    df = df.drop_duplicates(subset=['title_lower', 'company_lower'], keep='first', ignore_index=True)
    df = df.drop(columns=['title_lower', 'company_lower'])
    print(df.shape)
    df = df[~df['title'].str.contains('Senior', case=False)]
    df = df[~df['title'].str.contains('Manager', case=False)]
    df = df[~df['title'].str.contains('Director', case=False)]
    df = df[~df['title'].str.contains('Research', case=False)]
    df = df[~df['title'].str.contains('Principal', case=False)]
    df = df[~df['title'].str.contains('Lead', case=False)]
    df = df[~df['title'].str.contains('Intern', case=False)]
    df = df[~df['title'].str.contains('Co-Op', case=False)]
    df = df[~df['title'].str.contains('Professor', case=False)]
    df = df[~df['title'].str.contains('Sr.', case=False)]
    df = df[~df['title'].str.contains('President', case=False)]
    print("After dropping bad titles \n")
    print(df.shape)
    df = df.drop_duplicates(subset=['title', 'company', 'date_posted'])
    print(df.shape)
    df['location'] = df['location'].fillna('Remote, US')
    df['location'] = df['location'].str.replace('US', 'Remote, US')
    df['location'] = df['location'].str.replace('United States', 'Remote, US')
    df['location_split'] = df['location'].str.split(',').apply(lambda x: len(x) if x else 0)

    df = df[df['location_split'] > 1]
    print('after splitting location: \n')
    print(df.shape)
    df['city'] = df['location'].str.split(',').apply(lambda x: x[0].strip())
    df['state'] = df['location'].str.split(',').apply(lambda x: x[1].strip())
    df.drop(columns=['location', 'location_split'], inplace=True)

    df.loc[(df['state'] == 'US') & (df['city'] == 'Remote'), 'city'] = 'Remote'
    df.loc[(df['state'] == 'US') & (df['city'] == 'Remote'), 'state'] = 'Remote'

    print('after processing location: \n')
    print(df.shape)

    print('after processing location: \n')
    print(df.shape)

    df['cleaned_desc'] = df['description'].apply(extract_info)
    print('after processing bullet points: \n')
    print(df.shape)
    df['cleaned_desc_2_len'] = df['cleaned_desc'].apply(lambda x: len(x.split()))
    df = df[df['cleaned_desc_2_len'] > 20]
    df['description_clean'] = df['cleaned_desc'].apply(preprocess_text)

    
    return df