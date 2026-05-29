# 🔍 Azure AI Bias Monitoring Pipeline

A cloud-deployed end-to-end machine learning pipeline that detects and monitors 
social bias in AI-generated image models, built on Microsoft Azure.

## 🏗️ Architecture
[Python Data Pipeline]
↓
[Azure Blob Storage - Raw Data Lake]
↓
[Bias Detection - PCA + KMeans + DBSCAN]
↓
[Azure SQL Database - Results Storage]
↓
[Interactive Plotly Dashboard]


## 🎯 Project Overview

This project extends my MSc dissertation research on AI fairness into a 
production-ready cloud pipeline. It detects systemic gender, racial, and 
cultural bias in AI image generation models (Stable Diffusion, DALL·E, Gemini) 
without predefined labels, using unsupervised machine learning.

## 🔑 Key Findings

- **Gender Bias:** CEO, Doctor, and Engineer prompts generated 65-85% male representations
- **Racial Bias:** White representation dominant across high-status professions
- **Model Comparison:** Stable Diffusion showed highest average bias score (0.61)
- **Fairness-Utility Tradeoff:** High CLIP alignment scores consistently appeared alongside biased outputs
- **311 high-bias records** detected (bias score > 0.7) out of 1000 generated

## ☁️ Azure Services Used

| Service | Purpose |
|---------|---------|
| Azure Blob Storage | Raw data lake for generated image metadata |
| Azure SQL Database | Structured storage for processed bias results |
| Azure Resource Groups | Cloud resource organisation |

## 🛠️ Tech Stack

- **Cloud:** Microsoft Azure (Blob Storage, SQL Database)
- **ML/AI:** Scikit-learn (PCA, KMeans, DBSCAN), CLIP Embeddings
- **Data:** Python, Pandas, NumPy, SQLAlchemy
- **Visualisation:** Plotly (Interactive Dashboard)
- **Version Control:** Git, GitHub

## 📊 Dashboard Features

- Gender bias breakdown by profession
- Racial representation analysis
- AI model bias score comparison
- CLIP score vs bias score correlation
- PCA clustering visualisation
- High bias record identification

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/ASH0401-DSA/azure-ai-bias-monitor.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_CONTAINER=bias-data
AZURE_SQL_SERVER=your_server.database.windows.net
AZURE_SQL_DATABASE=biasmonitordb
AZURE_SQL_USERNAME=your_username
AZURE_SQL_PASSWORD=your_password

4. Run the pipeline
```bash
python src/generate_data.py
python src/bias_detection.py
python src/save_to_database.py
python src/dashboard.py
```

## 📁 Project Structure
azure-ai-bias-monitor/
├── src/
│   ├── generate_data.py      # Data generation & Azure upload
│   ├── bias_detection.py     # ML bias detection pipeline
│   ├── save_to_database.py   # Azure SQL integration
│   └── dashboard.py          # Interactive visualisation
├── data/                     # Local data storage
├── outputs/                  # Dashboard outputs
├── requirements.txt          # Dependencies
└── README.md

## 👤 Author

**Ashwin Nair**
- MSc Data Science & Analytics — Brunel University London
- [GitHub](https://github.com/ASH0401-DSA) 
- [LinkedIn](https://www.linkedin.com/in/ashwinnair2001)