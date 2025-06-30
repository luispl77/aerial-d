#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path
from google.cloud import storage
import vertexai
from vertexai.tuning import sft
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "aerial-bucket")  # Default to aerial-bucket if not specified
GCS_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gen-lang-client-0356477555-b5d578ff2efd.json")
PROJECT_ID = os.getenv("PROJECT_ID", "gen-lang-client-0356477555")
LOCATION = os.getenv("LOCATION", "us-central1")  # Default to us-central1 if not specified

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_PATH

def upload_file_to_gcs(local_path, gcs_path):
    """Upload a file to Google Cloud Storage."""
    try:
        # Initialize storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        # Upload the file
        blob.upload_from_filename(local_path)
        
        print(f"File {local_path} uploaded to gs://{GCS_BUCKET_NAME}/{gcs_path}")
        return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None

def start_training_job(gcs_jsonl_path):
    """Start a Gemini fine-tuning job using the uploaded JSONL file."""
    print(f"Initializing Vertex AI with project {PROJECT_ID} in {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    print(f"Starting fine-tuning job using {gcs_jsonl_path}")
    sft_tuning_job = sft.train(
        source_model="gemini-2.0-flash-lite-001",
        train_dataset=gcs_jsonl_path,
    )
    
    print("Fine-tuning job started. Polling for completion...")
    iteration = 0
    while not sft_tuning_job.has_ended:
        time.sleep(60)  # Check every minute
        sft_tuning_job.refresh()
        iteration += 1
        if iteration % 5 == 0:  # Print status every 5 minutes
            print(f"Job still running after {iteration} minutes...")
    
    print("Fine-tuning job completed!")
    print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
    print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")
    return sft_tuning_job

def main():
    parser = argparse.ArgumentParser(description="Upload data and start Gemini fine-tuning")
    parser.add_argument("--jsonl-path", default="gemini_triplet_finetuning_data/training_data.jsonl", 
                        help="Path to the JSONL file for training")
    args = parser.parse_args()
    
    # Validate required environment variables
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env file")
        return
    
    if not PROJECT_ID:
        print("Error: PROJECT_ID not found in .env file")
        return
    
    # Check if the JSONL file exists
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found at {jsonl_path}")
        return
    
    # Upload the JSONL file to GCS
    gcs_path = f"gemini_training/{jsonl_path.name}"
    gcs_jsonl_path = upload_file_to_gcs(str(jsonl_path), gcs_path)
    
    if not gcs_jsonl_path:
        print("Failed to upload JSONL file to GCS. Exiting.")
        return
    
    # Start the training job
    tuning_job = start_training_job(gcs_jsonl_path)
    
    # Save the model information to a file
    with open("tuned_model_info.txt", "w") as f:
        f.write(f"Tuned model name: {tuning_job.tuned_model_name}\n")
        f.write(f"Tuned model endpoint name: {tuning_job.tuned_model_endpoint_name}\n")
        f.write(f"Experiment: {tuning_job.experiment}\n")

if __name__ == "__main__":
    main() 