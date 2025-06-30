#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import vertexai
from vertexai.tuning import sft

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "gen-lang-client-0356477555")
LOCATION = os.getenv("LOCATION", "us-central1")
GCS_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gen-lang-client-0356477555-b5d578ff2efd.json")

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_PATH

# Initialize Vertex AI
print(f"Initializing Vertex AI with project {PROJECT_ID} in {LOCATION}")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# List tuning jobs
print("Listing all tuning jobs:")
tuning_jobs = sft.SupervisedTuningJob.list()

if not tuning_jobs:
    print("No tuning jobs found.")
else:
    for job in tuning_jobs:
        print(f"\nJob ID: {job.resource_name}")
        print(f"Status: {job.state}")
        print(f"Create Time: {job.create_time}")
        
        # For completed jobs, show the model information
        if job.has_ended and hasattr(job, 'tuned_model_name'):
            print(f"Tuned Model Name: {job.tuned_model_name}")
            if hasattr(job, 'tuned_model_endpoint_name'):
                print(f"Tuned Model Endpoint: {job.tuned_model_endpoint_name}") 