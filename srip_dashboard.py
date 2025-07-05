import streamlit as st 
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("filtered_evaluation_with_flags.csv")

df = load_data()
st.title("SRIP Metrics Dashboard: Filtered Answer Evaluation")
st.markdown("This dashboard helps explore the quality of the filtered student answers using various NLP metrics.")

# ===============================
# Filters
# ===============================
st.sidebar.header("Filter Options")

# Metric sliders
sts_min, sts_max = st.sidebar.slider("STS Score Range", 0.0, 1.0, (0.0, 1.0))
bert_min, bert_max = st.sidebar.slider("BERTScore F1 Range", 0.0, 1.0, (0.0, 1.0))
edit_min, edit_max = st.sidebar.slider("Edit Distance Range", 0, int(df["Edit_Distance"].max()), (0, int(df["Edit_Distance"].max())))

# Flag filters (checkboxes)
st.sidebar.markdown("### Flag Filters")
flag_filters = {
    "Only Good STS": st.sidebar.checkbox("Flag_STS = Good", False),
    "Only Good BERT": st.sidebar.checkbox("Flag_BERT = Good", False),
    "Only Ideal Compression": st.sidebar.checkbox("Flag_Compression = Ideal", False),
    "Only Entailed NLI": st.sidebar.checkbox("Flag_NLI = Entailed", False),
}

# Apply metric filters
filtered_df = df[
    (df["STS_score"].between(sts_min, sts_max)) &
    (df["BERTScore_F1"].between(bert_min, bert_max)) &
    (df["Edit_Distance"].between(edit_min, edit_max))
]

# Apply flag-based filters
if flag_filters["Only Good STS"]:
    filtered_df = filtered_df[filtered_df["Flag_STS"] == "Good"]
if flag_filters["Only Good BERT"]:
    filtered_df = filtered_df[filtered_df["Flag_BERT"] == "Good"]
if flag_filters["Only Ideal Compression"]:
    filtered_df = filtered_df[filtered_df["Flag_Compression"] == "Ideal"]
if flag_filters["Only Entailed NLI"]:
    filtered_df = filtered_df[filtered_df["Flag_NLI"] == "Entailed"]

st.markdown(f"**Total entries after filtering:** {len(filtered_df)}")

# ===============================
# View Examples
# ===============================
st.subheader("Explore Student vs. Filtered Answers")

if len(filtered_df) > 0:
    selected_index = st.selectbox("Choose a row to view", filtered_df.index)
    entry = filtered_df.loc[selected_index]

    st.write(f"**Question**: {entry['question']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Student Answer**")
        st.code(entry["student_answer"], language="markdown")
    with col2:
        st.markdown("**Filtered Answer**")
        st.code(entry["filtered_answer"], language="markdown")
else:
    st.warning("No entries match the current filter selection.")

# ===============================
# Metric Distribution Charts
# ===============================
st.subheader("Metric Distributions")

metric = st.selectbox("Choose a metric to view distribution", [
    "STS_score", "BERTScore_F1", "Compression_ratio", "Edit_Distance", "Normalized_Edit_Distance"
])

fig1 = px.histogram(df, x=metric, nbins=30, title=f"Distribution of {metric}", marginal="box")
st.plotly_chart(fig1, use_container_width=True)

# ===============================
# Correlation Heatmap
# ===============================
# st.subheader("Metric Correlation Heatmap")

metrics_cols = [
    "STS_score", "BERTScore_F1", "Compression_ratio",
    "Edit_Distance", "Normalized_Edit_Distance",
    "NLI_entail", "NLI_contradict", "NLI_neutral"
]

# corr_df = df[metrics_cols].corr()

# fig2, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
# st.pyplot(fig2)

# ===============================
# Summary Stats
# ===============================
st.subheader("Summary Statistics")
st.dataframe(df[metrics_cols].describe().T)

# ===============================
# NLI Label Distribution
# ===============================
st.subheader("NLI Label Distribution (All Answers Combined)")

nli_counts = df[["NLI_entail", "NLI_contradict", "NLI_neutral"]].sum().sort_values(ascending=False)
nli_fig = px.bar(nli_counts, x=nli_counts.index, y=nli_counts.values, labels={'x': 'NLI Class', 'y': 'Total Sentences'})
st.plotly_chart(nli_fig, use_container_width=True)

# ===============================
# Flag Counts (String Frequencies)
# ===============================
st.subheader("Flag Breakdown by Category")

for col in ["Flag_STS", "Flag_BERT", "Flag_Compression", "Flag_NLI"]:
    st.write(f"**{col} Distribution**")
    counts = df[col].value_counts().to_frame().rename(columns={col: "Count"})
    st.dataframe(counts)
