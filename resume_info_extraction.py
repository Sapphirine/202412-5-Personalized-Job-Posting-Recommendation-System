from pypdf import PdfReader
import json
from openai import OpenAI
import logging

# Set the logging level for 'pypdf' to ERROR or higher
logging.getLogger('pypdf').setLevel(logging.ERROR)


    # ATS extractor function (unchanged)
def resume_summary(pdf_path):
    api_key = "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"
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    text = page.extract_text()
    prompt = '''
        You are an AI bot designed to act as a professional for summarizing resumes. You are given a resume, 
        and your job is to summarize the following information from the resume to a maximum of 300 words:
        1. years of experience
        2. employment details
        3. technical skills,
        '''

    openai_client = OpenAI(
                api_key = api_key
            )

    messages=[
            {"role": "system",
            "content": prompt}
            ]

    user_content = text

    messages.append({"role": "user", "content": user_content})

    response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2500)

    data = response.choices[0].message.content

    return data

