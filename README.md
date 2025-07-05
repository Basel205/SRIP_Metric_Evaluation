# SRIP Metrics Dashboard

This Streamlit web app visualizes evaluation metrics for student answers before and after filtering, as part of a Summer Research Internship Project (SRIP).

### 🔍 Features
- Compare original vs. filtered answers
- Visualize key metrics (e.g., BERTScore, STS, edit distance, compression)
- Interactive filtering and flag-based views
- Based on precomputed dataset

### 📁 Files
- `srip_dashboard.py`: Main Streamlit app
- `filtered_evaluation_with_flags.csv`: Final dataset with all computed metrics and flags
- `requirements.txt`: Python dependencies for running the app

### 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run srip_dashboard.py