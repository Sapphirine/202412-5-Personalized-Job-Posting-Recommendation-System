import pandas as pd
import numpy as np
import re 


# def extract_filtered_bullet_texts(job_description):

#     # Define keywords to filter out
#     excluded_keywords = ['equal', 'disability', 'veteran', 'criminal', 'e\-verify', '401(k)', 'insurance', '----']

#     # Regex to match bullet points and the text following them
#     bullet_texts = re.findall(r'^\s*[\*\-\•\d+\.]\s*(.+)', job_description, re.MULTILINE)

#     # Filter out bullet points containing excluded keywords (case-insensitive)
#     filtered_bullets = [
#         text.strip() for text in bullet_texts
#         if not any(keyword in text.lower() for keyword in excluded_keywords)
#     ]
#     # if a sentence starts with or end with *, delete the sentence        
#     filtered_bullets = [x for x in filtered_bullets if not x.startswith('*') and not x.endswith('*')]
#     # if a sentence only contains numbers, delete the sentence
#     filtered_bullets = [x for x in filtered_bullets if not x.isdigit()]
#     # if a sentence only contains 2 or fewer words, delete the sentence
#     filtered_bullets = [x for x in filtered_bullets if len(x.split()) > 2]

#     return filtered_bullets

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
                               'equal', 'do', 'office', 'offices', 'why', 'interview', 'opportunity', 'about', 'perks', 'fits and pe']

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
    

    return extracted_data

def clean_data(df):
    df = df[['id', 'site', 'job_url', 'title', 'company', 'location', 'date_posted', 'description', 'date_fetched']]
    df = df.dropna(subset=['description', 'title', 'company'])
    df = df[~df['title'].str.contains('Senior', case=False)]
    df = df[~df['title'].str.contains('Manager', case=False)]
    df = df[~df['title'].str.contains('Director', case=False)]
    df = df[~df['title'].str.contains('Research', case=False)]
    df = df[~df['title'].str.contains('Principal', case=False)]
    df = df[~df['title'].str.contains('Lead', case=False)]
    df = df[~df['title'].str.contains('Intern', case=False)]
    df = df[~df['title'].str.contains('Co-Op', case=False)]
    df = df.drop_duplicates(subset=['title', 'company', 'date_posted'])
    df['location'] = df['location'].fillna('')  # Fill NaN values with empty string
    df['location_split'] = df['location'].str.split(',').apply(lambda x: len(x) if x else 0)

    df = df[df['location_split'] > 1]
    df['city'] = df['location'].str.split(',').apply(lambda x: x[0].strip())
    df['state'] = df['location'].str.split(',').apply(lambda x: x[1].strip())
    df.drop(columns=['location', 'location_split'], inplace=True)

    df = df[~(df['state'] == 'US') & (df['city'] != 'Remote')]
    df = df[(df['description'].str.contains('Qualifications', case=False)) 
        | (df['description'].str.contains('Skills', case=False))
        | (df['description'].str.contains('Job Functions', case=False))
        | (df['description'].str.contains('Responsibilities', case=False))
        | (df['description'].str.contains('Requirements', case=False))]


    df['cleaned_desc'] = df['description'].apply(extract_info)
    df['cleaned_desc_2_len'] = df['cleaned_desc'].apply(lambda x: len(x.split()))
    df = df[df['cleaned_desc_2_len'] > 100]
    df = df[df['cleaned_desc_2_len'] < 1000]

    
    return df