import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load pretrained BERT model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Load trained logistic regression model ---
with open("bert_model.pkl", "rb") as f:
    clf = pickle.load(f)

# --- Regex rules for structured logs ---
regex_rules = {
    'System Notification': r'(uploaded successfully|system notification|user logged in)',
    'User Action': r'(deleted|created|updated|performed action|login|logout)',
    'Deprecation Warning': r'(deprecated|will be removed|obsolete)'
}

# --- Functions ---
def clean_log_message(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\-_:\/]', '', text)
    return text.strip()

def apply_regex_rules(log_msg, current_pred):
    for label, pattern in regex_rules.items():
        if re.search(pattern, log_msg, re.IGNORECASE):
            return label
    return current_pred

# Color-coded labels
label_colors = {
    'Critical Error': 'background-color: #ff4d4d; color: white',
    'Error': 'background-color: #ff9999; color: black',
    'HTTP Status': 'background-color: #66b3ff; color: black',
    'Resource Usage': 'background-color: #99ff99; color: black',
    'Security Alert': 'background-color: #ffcc66; color: black',
    'System Notification': 'background-color: #c2f0c2; color: black',
    'User Action': 'background-color: #ffb3e6; color: black',
    'Deprecation Warning': 'background-color: #cccccc; color: black'
}

def color_labels(val):
    return label_colors.get(val, '')

# Color-coded prediction source
source_colors = {
    'ML': 'background-color: #80b3ff; color: white',      # blue
    'Regex': 'background-color: #66ff66; color: black'    # green
}

def color_source(val):
    return source_colors.get(val, '')

# --- Streamlit UI ---
st.title("Hybrid Log Classification System")
st.markdown("Paste your log messages below (one per line) and click **Predict**.")

log_text = st.text_area("Paste log messages here:")

if st.button("Predict"):
    if log_text:
        # Split logs into list
        log_list = log_text.strip().split("\n")
        df = pd.DataFrame({'log_message': log_list})

        # Clean logs
        df['log_message_clean'] = df['log_message'].apply(clean_log_message)

        # Generate embeddings
        embeddings = model.encode(df['log_message_clean'].tolist(), show_progress_bar=True)

        # ML predictions
        y_pred = clf.predict(embeddings)

        # Apply regex rules
        df['predicted_label'] = [apply_regex_rules(msg, pred) for msg, pred in zip(df['log_message_clean'], y_pred)]

        # Identify prediction source
        df['prediction_source'] = [
            'Regex' if apply_regex_rules(msg, pred) != pred else 'ML'
            for msg, pred in zip(df['log_message_clean'], y_pred)
        ]

        # --- Display color-coded table ---
        st.subheader("Predicted Labels and Source (Color-coded)")
        st.dataframe(
            df[['log_message', 'predicted_label', 'prediction_source']].style
              .applymap(color_labels, subset=['predicted_label'])
              .applymap(color_source, subset=['prediction_source'])
        )

        # --- Prediction Source Distribution (Pie Chart) ---
        st.subheader("Prediction Source Distribution")
        source_counts = df['prediction_source'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(7,7))
        ax1.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', colors=['#80b3ff', '#66ff66'], radius=0.7)
        ax1.set_title("ML vs Regex Predictions")
        st.pyplot(fig1)

        # --- Predicted Label Distribution (Bar Chart) ---
        st.subheader("Predicted Label Distribution")
        label_counts = df['predicted_label'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(10,5))
        sns.barplot(x=label_counts.index, y=label_counts.values, palette='tab10', ax=ax2)
        ax2.set_ylabel("Number of Logs")
        ax2.set_xlabel("Predicted Labels")
        ax2.set_title("Count of Logs per Predicted Label")
        plt.xticks(rotation=45)
        for i, v in enumerate(label_counts.values):
            ax2.text(i, v + 0.5, str(v), ha='center')
        st.pyplot(fig2)

        # --- Prediction Source per Label (Stacked Bar Chart) ---
        st.subheader("Prediction Source per Label")
        source_label_counts = df.groupby(['predicted_label', 'prediction_source']).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(10,6))
        source_label_counts.plot(kind='bar', stacked=True, ax=ax3, color=['#80b3ff', '#66ff66'])
        ax3.set_ylabel("Number of Logs")
        ax3.set_xlabel("Predicted Labels")
        ax3.set_title("Number of Logs by Prediction Source (ML vs Regex)")
        plt.xticks(rotation=45)
        for i, row in enumerate(source_label_counts.values):
            for j, val in enumerate(row):
                if val > 0:
                    ax3.text(i, val/2 + sum(row[:j]), str(val), ha='center', color='white', fontweight='bold')
        plt.legend(title='Prediction Source')
        st.pyplot(fig3)

        # --- Download Predictions ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="predicted_logs.csv", mime='text/csv')

    else:
        st.warning("Please paste some logs first!")
