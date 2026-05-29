import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from dotenv import load_dotenv
import os
import io

# Load environment variables
load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

# Download data from Azure Blob Storage
def download_latest_blob():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # Get latest blob
    blobs = list(container_client.list_blobs())
    latest_blob = sorted(blobs, key=lambda x: x.last_modified, reverse=True)[0]
    
    print(f"Downloading: {latest_blob.name}")
    blob_client = container_client.get_blob_client(latest_blob.name)
    data = blob_client.download_blob().readall()
    
    df = pd.read_csv(io.BytesIO(data))
    print(f"Downloaded {len(df)} records")
    return df

# Prepare features for clustering
def prepare_features(df):
    le = LabelEncoder()
    
    df['gender_encoded'] = le.fit_transform(df['gender_representation'])
    df['race_encoded'] = le.fit_transform(df['racial_representation'])
    df['prompt_encoded'] = le.fit_transform(df['prompt'])
    df['model_encoded'] = le.fit_transform(df['model'])
    
    features = ['gender_encoded', 'race_encoded', 
                'clip_alignment_score', 'bias_score',
                'prompt_encoded', 'model_encoded']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df

# Run PCA
def run_pca(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return X_pca

# Run KMeans clustering
def run_kmeans(X_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    print(f"KMeans clusters: {np.unique(clusters)}")
    return clusters

# Run DBSCAN clustering
def run_dbscan(X_scaled):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    print(f"DBSCAN clusters found: {len(np.unique(clusters))}")
    return clusters

# Calculate bias metrics
def calculate_bias_metrics(df):
    print("\n--- BIAS ANALYSIS REPORT ---")
    
    # Gender bias by prompt
    print("\nGender representation by prompt:")
    gender_bias = df.groupby(['prompt', 'gender_representation']).size().unstack(fill_value=0)
    print(gender_bias)
    
    # Racial bias by prompt
    print("\nRacial representation by prompt:")
    race_bias = df.groupby(['prompt', 'racial_representation']).size().unstack(fill_value=0)
    print(race_bias)
    
    # Bias score by model
    print("\nAverage bias score by model:")
    model_bias = df.groupby('model')['bias_score'].mean()
    print(model_bias)
    
    # High bias records
    high_bias = df[df['bias_score'] > 0.7]
    print(f"\nHigh bias records (score > 0.7): {len(high_bias)}")
    
    return gender_bias, race_bias, model_bias

# Upload results back to Azure
def upload_results(df):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    csv_data = df.to_csv(index=False)
    blob_client = container_client.get_blob_client("bias_results.csv")
    blob_client.upload_blob(csv_data, overwrite=True)
    print("\nResults uploaded to Azure Blob Storage!")

if __name__ == "__main__":
    print("Starting bias detection pipeline...\n")
    
    # Download data
    df = download_latest_blob()
    
    # Prepare features
    X_scaled, df = prepare_features(df)
    
    # Run PCA
    print("\nRunning PCA...")
    X_pca = run_pca(X_scaled)
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]
    
    # Run clustering
    print("\nRunning KMeans clustering...")
    df['kmeans_cluster'] = run_kmeans(X_scaled)
    
    print("\nRunning DBSCAN clustering...")
    df['dbscan_cluster'] = run_dbscan(X_scaled)
    
    # Calculate bias metrics
    calculate_bias_metrics(df)
    
    # Upload results
    upload_results(df)
    
    print("\nBias detection pipeline complete!")