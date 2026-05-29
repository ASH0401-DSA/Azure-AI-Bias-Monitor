import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Azure connection
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

# Simulate AI image generation bias data
def generate_bias_data(n_samples=1000):
    np.random.seed(42)
    
    prompts = [
        "a doctor", "an engineer", "a nurse", "a CEO",
        "a teacher", "a scientist", "a lawyer", "a chef"
    ]
    
    genders = ["male", "female", "neutral"]
    races = ["white", "black", "asian", "hispanic", "middle_eastern"]
    models = ["stable_diffusion", "dalle", "gemini"]
    
    data = []
    for i in range(n_samples):
        prompt = np.random.choice(prompts)
        model = np.random.choice(models)
        
        # Simulate bias - doctors/engineers skewed male/white
        if prompt in ["a doctor", "an engineer", "a CEO", "a scientist"]:
            gender = np.random.choice(genders, p=[0.65, 0.30, 0.05])
            race = np.random.choice(races, p=[0.60, 0.10, 0.15, 0.10, 0.05])
        else:
            gender = np.random.choice(genders, p=[0.35, 0.55, 0.10])
            race = np.random.choice(races, p=[0.40, 0.20, 0.20, 0.15, 0.05])
        
        clip_score = np.random.uniform(0.6, 0.95)
        bias_score = np.random.uniform(0.3, 0.9)
        
        data.append({
            "id": i + 1,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": model,
            "gender_representation": gender,
            "racial_representation": race,
            "clip_alignment_score": round(clip_score, 4),
            "bias_score": round(bias_score, 4)
        })
    
    return pd.DataFrame(data)

# Upload to Azure Blob Storage
def upload_to_blob(df):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # Save as CSV
    csv_data = df.to_csv(index=False)
    blob_name = f"bias_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)
    
    print(f"Successfully uploaded {blob_name} to Azure Blob Storage!")
    print(f"Total records: {len(df)}")
    return blob_name

if __name__ == "__main__":
    print("Generating bias data...")
    df = generate_bias_data(1000)
    print(f"Generated {len(df)} records")
    print(df.head())
    
    print("\nUploading to Azure Blob Storage...")
    blob_name = upload_to_blob(df)
    print("Done!")