#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import argparse
import datetime
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions

# Constants
PROJECT_ID = "564504826453"
LOCATION = "us-central1"
BUCKET_NAME = "aerial-bucket"
MODEL_ID = "gemini-2.0-flash-001"
DEFAULT_DATASET_FOLDER = "dataset"
DEFAULT_OUTPUT_PATH = f"gs://{BUCKET_NAME}/batch_inference/results"

def parse_args():
    parser = argparse.ArgumentParser(description='Run batch prediction for LLM processing')
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both',
                      help='Which split to process (train, val, or both)')
    parser.add_argument('--dataset_folder', type=str, default=DEFAULT_DATASET_FOLDER,
                      help=f'Dataset folder name (default: {DEFAULT_DATASET_FOLDER})')
    parser.add_argument('--input_jsonl_train', type=str,
                      help='Path to input JSONL file for training split')
    parser.add_argument('--input_jsonl_val', type=str,
                      help='Path to input JSONL file for validation split')
    parser.add_argument('--output_gcs', type=str,
                      help='GCS path for output results')
    return parser.parse_args()

def log_with_timestamp(message):
    """Add timestamp to log messages"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def download_results_from_gcs(job_name, split, results_dir):
    """Download results from GCS to local directory"""
    try:
        from google.cloud import storage
        
        # Initialize storage client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Create local directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Get the export location from the job
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            http_options=HttpOptions(api_version="v1")
        )
        job = client.batches.get(name=job_name)
        
        # Try different ways to get the export location
        export_location = None
        if hasattr(job, 'export_location') and job.export_location:
            export_location = job.export_location
        elif hasattr(job, 'output_location') and job.output_location:
            export_location = job.output_location
        elif hasattr(job, 'output_uri') and job.output_uri:
            export_location = job.output_uri
            
        if not export_location:
            log_with_timestamp("No export location found in job. Available attributes:")
            for attr in dir(job):
                if not attr.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(job, attr)
                        log_with_timestamp(f"  {attr}: {value}")
                    except:
                        pass
            
            # If we can't get the export location from the job, try to find the most recent results file
            log_with_timestamp("Attempting to find most recent results file...")
            prefix = f"batch_inference/results/{split}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                log_with_timestamp(f"No results found in {prefix}")
                return False
                
            # Sort blobs by update time, most recent first
            latest_blob = max(blobs, key=lambda x: x.updated if x.updated else datetime.min)
            log_with_timestamp(f"Found most recent results file: {latest_blob.name}")
            
            # Download the file
            local_path = os.path.join(results_dir, f"batch_inference_results_{split}.jsonl")
            log_with_timestamp(f"Downloading results from {latest_blob.name} to {local_path}")
            latest_blob.download_to_filename(local_path)
            
            log_with_timestamp(f"Successfully downloaded results to {local_path}")
            return True
            
        # If we have an export location, try to download from there
        local_path = os.path.join(results_dir, f"batch_inference_results_{split}.jsonl")
        
        # Try different possible paths for the predictions file
        possible_paths = [
            f"{export_location.replace(f'gs://{BUCKET_NAME}/', '')}/predictions.jsonl",
            f"batch_inference/results/{split}/predictions.jsonl",
            f"batch_inference/results/{split}/prediction-model-*/predictions.jsonl"
        ]
        
        for path in possible_paths:
            try:
                if "*" in path:
                    # Handle wildcard path by finding most recent matching file
                    prefix = path.split("*")[0]
                    blobs = list(bucket.list_blobs(prefix=prefix))
                    if blobs:
                        latest_blob = max(blobs, key=lambda x: x.updated if x.updated else datetime.min)
                        log_with_timestamp(f"Found most recent results file: {latest_blob.name}")
                        latest_blob.download_to_filename(local_path)
                        log_with_timestamp(f"Successfully downloaded results to {local_path}")
                        return True
                else:
                    blob = bucket.blob(path)
                    if blob.exists():
                        log_with_timestamp(f"Downloading results from {path} to {local_path}")
                        blob.download_to_filename(local_path)
                        log_with_timestamp(f"Successfully downloaded results to {local_path}")
                        return True
            except Exception as e:
                log_with_timestamp(f"Error trying path {path}: {e}")
                continue
        
        log_with_timestamp("Could not find or download results file from any expected location")
        return False
        
    except Exception as e:
        log_with_timestamp(f"Error downloading results: {e}")
        return False

def upload_jsonl_to_gcs(local_jsonl_path):
    """Upload the JSONL file to GCS if it's not already there"""
    if local_jsonl_path.startswith("gs://"):
        return local_jsonl_path
    
    # If we need to upload the file to GCS
    from google.cloud import storage
    
    # Initialize storage client
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Generate GCS path
    filename = os.path.basename(local_jsonl_path)
    blob_name = f"batch_inference/input/{filename}"
    blob = bucket.blob(blob_name)
    
    log_with_timestamp(f"Uploading {local_jsonl_path} to GCS...")
    blob.upload_from_filename(local_jsonl_path)
    
    gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
    log_with_timestamp(f"Uploaded to {gcs_uri}")
    
    return gcs_uri

async def run_batch_prediction_async(input_jsonl_gcs, output_gcs_path, split, results_dir):
    """Start and monitor a batch prediction job asynchronously"""
    # Initialize client with proper authentication for Vertex AI
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1")
    )
    
    log_with_timestamp(f"Starting batch prediction job for {split} split using:")
    log_with_timestamp(f"  Model: {MODEL_ID}")
    log_with_timestamp(f"  Input JSONL: {input_jsonl_gcs}")
    log_with_timestamp(f"  Output path: {output_gcs_path}")
    
    # Create batch job
    job = client.batches.create(
        model=MODEL_ID,
        src=input_jsonl_gcs,
        config=CreateBatchJobConfig(dest=output_gcs_path),
    )
    
    job_name = job.name
    log_with_timestamp(f"Job created with name: {job_name}")
    log_with_timestamp(f"Initial job state: {job.state}")
    
    # Define completed states
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }
    
    # Monitor job progress
    start_time = time.time()
    last_state = job.state
    
    try:
        while job.state not in completed_states:
            await asyncio.sleep(30)  # Check status every 30 seconds
            job = client.batches.get(name=job_name)
            
            # Only log if state changed
            if job.state != last_state:
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                log_with_timestamp(f"[{split}] Job state changed: {last_state} → {job.state} " 
                                  f"(Elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s)")
                last_state = job.state
            
            # Log progress every 5 minutes regardless of state change
            if time.time() - start_time > 0 and (time.time() - start_time) % 300 < 30:
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                log_with_timestamp(f"[{split}] Job still running: {job.state} "
                                  f"(Elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s)")
        
        # Job completed
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        log_with_timestamp(f"[{split}] Job completed with state: {job.state}")
        log_with_timestamp(f"[{split}] Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Show output location and download results if successful
        if job.state == JobState.JOB_STATE_SUCCEEDED:
            if hasattr(job, 'export_location') and job.export_location:
                log_with_timestamp(f"[{split}] Results available at: {job.export_location}")
            
            # If status is available, show it
            if hasattr(job, 'status') and job.status:
                log_with_timestamp(f"[{split}] Job status: {job.status}")
            
            # Download results
            if download_results_from_gcs(job_name, split, results_dir):
                return True
            else:
                log_with_timestamp(f"[{split}] Failed to download results, but job completed successfully")
                return True
        else:
            log_with_timestamp(f"[{split}] Job did not complete successfully. Final state: {job.state}")
            
            # If error is available, show it
            if hasattr(job, 'error') and job.error:
                log_with_timestamp(f"[{split}] Error: {job.error}")
            
            return False
            
    except Exception as e:
        log_with_timestamp(f"[{split}] Error monitoring job: {e}")
        log_with_timestamp(f"[{split}] Job may still be running. Job name: {job_name}")
        return False

async def process_split(split, input_jsonl, output_gcs, results_dir):
    """Process a single split asynchronously"""
    log_with_timestamp(f"Processing {split} split...")
    
    # If input is a local file, upload it to GCS first
    if not input_jsonl.startswith("gs://"):
        if not os.path.exists(input_jsonl):
            log_with_timestamp(f"Error: Input file {input_jsonl} not found")
            return False
        
        input_gcs_path = upload_jsonl_to_gcs(input_jsonl)
    else:
        input_gcs_path = input_jsonl
    
    # Set output path for this split
    split_output_path = f"{output_gcs}/{split}"
    
    # Run the batch prediction
    success = await run_batch_prediction_async(input_gcs_path, split_output_path, split, results_dir)
    
    if success:
        log_with_timestamp(f"Batch prediction completed successfully for {split} split")
    else:
        log_with_timestamp(f"Batch prediction did not complete successfully for {split} split")
    
    return success

async def main_async(split, input_jsonl_train, input_jsonl_val, output_gcs, results_dir):
    # Determine which splits to process
    tasks = []
    if split == 'both':
        tasks = [
            process_split('train', input_jsonl_train, output_gcs, results_dir),
            process_split('val', input_jsonl_val, output_gcs, results_dir)
        ]
    else:
        tasks = [process_split(split, input_jsonl_train if split == 'train' else input_jsonl_val, output_gcs, results_dir)]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Check if any task failed
    if not all(results):
        sys.exit(1)

def main():
    args = parse_args()
    dataset_folder = args.dataset_folder
    
    # Set default paths if not provided
    input_jsonl_train = args.input_jsonl_train or os.path.join(dataset_folder, "batch_prediction/batch_prediction_requests_train.jsonl")
    input_jsonl_val = args.input_jsonl_val or os.path.join(dataset_folder, "batch_prediction/batch_prediction_requests_val.jsonl")
    output_gcs = args.output_gcs or DEFAULT_OUTPUT_PATH
    results_dir = os.path.join(dataset_folder, "batch_prediction")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the async main function
    asyncio.run(main_async(args.split, input_jsonl_train, input_jsonl_val, output_gcs, results_dir))

if __name__ == "__main__":
    main() 