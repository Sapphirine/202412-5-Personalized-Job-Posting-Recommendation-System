import os
import pandas as pd
from datetime import date
from jobspy import scrape_jobs

def update_job_data(job_titles = ["software engineer", "data scientist", "data analyst", "sales", "human resources"], 
                    location = "United States",
                    hours_old = 720, 
                    results_per_query = 40, 
                    total_results_wanted = 3000, output_file="job_data.csv"):

    all_jobs = []
    max_queries = total_results_wanted // results_per_query  # Number of pages to fetch

    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
    else:
        existing_data = pd.DataFrame()

    for title in job_titles:
        print(f"Searching for '{title}' in '{location}'...")
        for page in range(max_queries):
            try:
                # Scrape job postings
                jobs = scrape_jobs(
                    site_name=["indeed", "zip_recruiter", "glassdoor"],
                    search_term=title,
                    location=location,
                    results_wanted=results_per_query,
                    hours_old=hours_old,
                    offset=page * results_per_query,
                )

                if jobs.empty:
                    print(f"No jobs found on page {page + 1}")
                    break

                all_jobs.extend(jobs.to_dict('records'))  
                print(f"Fetched {len(jobs)} jobs for page {page + 1}")

            except Exception as e:
                print(f"Error fetching page {page + 1} for '{title}' in '{location}': {e}")
                break
    new_data = pd.DataFrame(all_jobs)
    today = date.today()
    today = today.strftime("%Y-%m-%d")
    new_data["date_fetched"] = today

    if not new_data.empty:
        new_data = new_data.drop_duplicates()
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates()
        combined_data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    else:
        print("No new data fetched.")
