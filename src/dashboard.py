import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER")

# Download results from Blob Storage
def download_results():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client("bias_results.csv")
    data = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(data))
    print(f"Downloaded {len(df)} records")
    return df

def create_dashboard(df):
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Gender Bias by Profession",
            "Racial Bias by Profession",
            "Bias Score by AI Model",
            "CLIP Score vs Bias Score",
            "PCA Clustering Visualisation",
            "High Bias Records by Profession"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Gender bias by profession
    gender_bias = df.groupby(['prompt', 'gender_representation']).size().reset_index(name='count')
    for gender in gender_bias['gender_representation'].unique():
        data = gender_bias[gender_bias['gender_representation'] == gender]
        fig.add_trace(
            go.Bar(name=gender, x=data['prompt'], y=data['count'], showlegend=True),
            row=1, col=1
        )

    # 2. Racial bias by profession
    race_bias = df.groupby(['prompt', 'racial_representation']).size().reset_index(name='count')
    for race in race_bias['racial_representation'].unique():
        data = race_bias[race_bias['racial_representation'] == race]
        fig.add_trace(
            go.Bar(name=race, x=data['prompt'], y=data['count'], showlegend=True),
            row=1, col=2
        )

    # 3. Bias score by model
    model_bias = df.groupby('model')['bias_score'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=model_bias['model'],
            y=model_bias['bias_score'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. CLIP score vs Bias score scatter
    fig.add_trace(
        go.Scatter(
            x=df['clip_alignment_score'],
            y=df['bias_score'],
            mode='markers',
            marker=dict(
                color=df['kmeans_cluster'],
                colorscale='Viridis',
                size=4,
                opacity=0.6
            ),
            showlegend=False
        ),
        row=2, col=2
    )

    # 5. PCA clustering
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for cluster in df['kmeans_cluster'].unique():
        cluster_data = df[df['kmeans_cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['pca_x'],
                y=cluster_data['pca_y'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(size=4, opacity=0.6, color=colors[cluster]),
                showlegend=True
            ),
            row=3, col=1
        )

    # 6. High bias records by profession
    high_bias = df[df['bias_score'] > 0.7].groupby('prompt').size().reset_index(name='count')
    fig.add_trace(
        go.Bar(
            x=high_bias['prompt'],
            y=high_bias['count'],
            marker_color='#FF6B6B',
            showlegend=False
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        height=1200,
        title=dict(
            text="🔍 AI Bias Monitoring Dashboard — Azure Cloud Pipeline",
            font=dict(size=24),
            x=0.5
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='white'),
        barmode='group'
    )

    # Save dashboard
    fig.write_html("outputs/bias_dashboard.html")
    print("Dashboard saved to outputs/bias_dashboard.html!")
    
    # Show in browser
    fig.show()

if __name__ == "__main__":
    print("Building dashboard from Azure data...\n")
    df = download_results()
    create_dashboard(df)
    print("Dashboard complete!")