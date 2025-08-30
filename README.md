# Hybrid Log Classification System 🖥️

This is a hybrid ML-powered log classification system that categorizes logs from multiple sources. It combines **BERT embeddings**, **logistic regression**, **regex-based rules**, and optional **LLM pattern detection** for structured and unstructured logs.

Users can paste log messages into a text box and receive **color-coded predicted labels** along with visual analytics.

---

## Features

- Hybrid ML + Regex log classification  
- Color-coded predictions for easy understanding  
- Charts for prediction source, label distribution, and stacked view of ML vs Regex  
- Download predictions as CSV  
- Supports multi-line log input via text box  

---

## Project Structure

```
/hybrid-log-classifier
├── dataset/
│   ├── mydataset.csv
├── app.py                                   # Streamlit app
├── bert_model.pkl                           # Trained logistic regression model
├── log_classifier.ipynb                     # Jupyter notebook
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Tech Stack

- **ML & Embeddings:** BERT (SentenceTransformers), Logistic Regression  
- **Data Analysis:** Pandas, Scikit-learn, DBSCAN clustering  
- **Regex-based Classification:** For structured log patterns  
- **Visualization & UI:** Streamlit, Matplotlib, Seaborn  
- **Optional Conceptual LLM Integration:** For advanced pattern detection  

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hybrid-log-classifier.git
cd hybrid-log-classifier
```

### 2. Create Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt

```

### 3. Place Trained Model

```bash
bert_model.pkl
```

### Running the app

```bash
streamlit run app.py
```

- The app will open in your browser
- Paste logs in the text box and click Predict

---

## Streamlit Features

### Log Input

- Users paste multiple log messages (one per line).
- Logs are preprocessed and cleaned for ML inference.

### Predicted Labels Table

- Displays predicted labels with color-coded highlighting.
- Shows prediction source (ML vs Regex).

### Visualizations

- Prediction Source Pie Chart – % of logs classified by ML vs Regex
- Predicted Label Distribution Bar Chart – frequency of each label
- Prediction Source per Label Stacked Bar Chart – ML vs Regex contribution for each label

### Download

- Users can download the predictions as a CSV file.

---

### Sample Log Labels
- Critical Error
- Error
- HTTP Status
- Resource Usage
- Security Alert
- System Notification
- User Action
- Deprication Warning

---

## Model Performance

- **Overall Accuracy:** ~91% on the test set
- Works best for frequent log types like `HTTP Status`, `Error`, and `Critical Error`.
- Rare or ambiguous logs (e.g., `User Action`, `Deprecation Warning`) may have lower prediction reliability.

---

## Future Enhancements

- Integrate actual LLM-based inference for advanced pattern detection
- Add file upload feature for large log datasets
- Add authentication and multi-user support
- Improve charts with interactivity
- Export charts and predictions to PDF

---

