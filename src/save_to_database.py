import pandas as pd
from azure.storage.blob import BlobServiceClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import io
import urllib

# Load environment variables
load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")
server = os.getenv("AZURE_SQL_SERVER")
database = os.getenv("AZURE_SQL_DATABASE")
username = os.getenv("AZURE_SQL_USERNAME")
password = os.getenv("AZURE_SQL_PASSWORD")

# Download results from Blob Storage
def download_results():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    blob_client = container_client.get_blob_client("bias_results.csv")
    data = blob_client.download_blob().readall()
    
    df = pd.read_csv(io.BytesIO(data))
    print(f"Downloaded {len(df)} records from Blob Storage")
    return df

# Connect to Azure SQL
def get_db_engine():
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    return engine

# Create table and save data
def save_to_sql(df, engine):
    # Create table if not exists
    create_table_sql = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='bias_results' AND xtype='U')
    CREATE TABLE bias_results (
        id INT,
        timestamp NVARCHAR(50),
        prompt NVARCHAR(100),
        model NVARCHAR(50),
        gender_representation NVARCHAR(50),
        racial_representation NVARCHAR(50),
        clip_alignment_score FLOAT,
        bias_score FLOAT,
        pca_x FLOAT,
        pca_y FLOAT,
        kmeans_cluster INT,
        dbscan_cluster INT
    )
    """
    
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
        print("Table created/verified successfully!")
    
    # Save dataframe to SQL
    df.to_sql('bias_results', engine, if_exists='replace', index=False)
    print(f"Saved {len(df)} records to Azure SQL Database!")

if __name__ == "__main__":
    print("Starting database upload...\n")
    
    # Download results
    df = download_results()
    
    # Connect to database
    print("Connecting to Azure SQL Database...")
    engine = get_db_engine()
    
    # Save to SQL
    save_to_sql(df, engine)
    
    print("\nDatabase upload complete!")