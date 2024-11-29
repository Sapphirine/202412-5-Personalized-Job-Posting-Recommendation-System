from pypdf import PdfReader
import json
from openai import OpenAI

# OpenAI API Key
api_key = "generate_your_own_api_key"

def resume_parser(pdf_path, output_file="parsed_resume.json"):
    """
    Parses a resume PDF and extracts structured information, saving the results to a JSON file.

    Args:
        pdf_path (str): Path to the PDF resume.
        output_file (str): Output JSON file to save the parsed data.
        
    Returns:
        None
    """
    # Read PDF and extract text
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    text = page.extract_text()

    # ATS extractor function (unchanged)
    def ats_extractor(resume_data):
        prompt = '''
        You are an AI bot designed to act as a professional for parsing resumes. You are given a resume, 
        and your job is to extract the following information from the resume:
        1. full name
        2. email id
        3. years of experience
        4. employment details
        5. technical skills, separated by commas.
        Give the extracted information in python dictionary format only, with keys being the above-mentioned points and values being the extracted information.
        '''

        openai_client = OpenAI(
                api_key = api_key
            )

        messages=[
            {"role": "system",
            "content": prompt}
            ]

        user_content = resume_data

        messages.append({"role": "user", "content": user_content})

        response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2500)

        data = response.choices[0].message.content

        return data

    # Clean and parse JSON function (unchanged)
    def clean_and_parse_json(text):
        try:
            text = text.strip()
            text = text.replace("“", '"').replace("”", '"')
            parsed_data = json.loads(text)
            return parsed_data
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None

    # Extract parsed data
    data = ats_extractor(text)
    parsed_data = clean_and_parse_json(data)

    if parsed_data is None:
        print("Failed to parse data. Exiting.")
        return

    # Construct employment experiences and technical skills
    employment_experiences = ""
    employment_experiences += "Year of experience: "
    employment_experiences += parsed_data['years of experience'] + ". "
    for detail in parsed_data['employment details']:
        for position in detail['position']:
            employment_experiences += position
        employment_experiences += ", "
        for responsibility in detail['responsibilities']:
            employment_experiences += responsibility

    technical_skills = parsed_data['technical skills']

    employment_experiences += "Technical skills: "
    employment_experiences += technical_skills

    technical_skills = technical_skills.split(',')
    technical_skills = [skill.strip() for skill in technical_skills]

    # Save parsed data to JSON file
    final_data = {
        "parsed_data": parsed_data,
        "employment_experiences": employment_experiences,
        "technical_skills": technical_skills
    }

    with open(output_file, "w") as outfile:
        json.dump(final_data, outfile, indent=4)

    print(f"Parsed data saved to {output_file}")

# Example usage:
# resume_parser("path_to_resume.pdf")
