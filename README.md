# DataMind AI — Premium CSV Analytics Agent & Dashboard

DataMind AI is a production-quality, modern AI-powered CSV Data Analyst Agent platform that allows users to upload datasets, clean them automatically, view key overview metrics, generate interactive responsive visualizations, create AI-powered business intelligence reports, and ask natural language questions about their datasets.

Designed with a premium SaaS-style dark mode dashboard aesthetic featuring crimson red accents and glassmorphic cards, DataMind AI brings advanced business intelligence directly to operations and executives.

---

## 🚀 Key Features

1. **CSV Upload & Demo System**
   - Drag-and-drop file uploader (supports up to 200MB).
   - "Load Sample Sales Dataset" simulator to test and demo the platform instantly.

2. **Automated Data Cleaning Pipeline**
   - Duplicate rows detection and removal.
   - Missing value imputation using multiple numerical (Median, Mean, Mode, Zero, Drop) and categorical (Mode, 'Unknown', Drop) strategies.
   - Type coercion and string parsing (cleans currency symbols, whitespace, commas, and formatting noises).

3. **Interactive & Adaptive Visualizations**
   - Automatically recommends up to 3 Plotly charts tailored to your dataset columns (time trends, categorical aggregations, scatter relationships, histograms).
   - Build custom charts dynamically with custom X-axis, Y-axis, and grouping columns.
   - Pearson numerical variable correlation heatmap.
   - Export interactive charts offline as standalone HTML packages.

4. **AI Strategic Insight Generator**
   - Powered by Google Gemini 1.5.
   - Summarizes dataset properties, metadata, and correlations into rich prompts.
   - Generates boardroom-ready Executive Summary, Key Business Insights, Trend Analysis, Data Risks & Outliers, and Concrete Actionable Recommendations.

5. **Natural Language CSV Chatbot**
   - Query datasets using raw English sentences (e.g. *"Show top 5 cities by profit margin"*).
   - Converts natural language queries into Pandas execution statements.
   - **Self-Correcting Execution Engine**: If the generated Pandas script errors out, the agent feeds the traceback back to the LLM to write a corrected version (up to 2 correction attempts).
   - Full history log with optional click-to-expand execution scripts for absolute transparency.

6. **Premium Report Exporters**
   - Export cleaned datasets back to CSV.
   - Export AI Business reports as raw Markdown.
   - Export premium, standalone, styled HTML executive reports with high-end typography and themes matching the online platform.

---

## 🛠️ Project Structure

```
datamind-ai/
│
├── app.py                  # Streamlit dashboard interface & state controller
├── requirements.txt        # Python package dependencies
├── README.md               # Product documentation
├── .env                    # Environment settings (Gemini API key)
│
├── data/                   # Directory to store uploaded datasets
├── reports/                # Directory for generated offline reports
├── charts/                 # Directory for exported figure logs
│
└── utils/
    ├── cleaner.py          # Data cleaning logic (nulls, duplicates, string noises)
    ├── analyzer.py         # Summary statistics, classifications, correlation matrix
    ├── visualizer.py       # Plotly chart styling & auto-chart matching engine
    ├── insights.py         # Gemini API Integration for deep analytics reports
    ├── chatbot.py          # Chatbot logic with self-healing pandas code runner
    └── exporter.py         # CSV, markdown, and SaaS-themed HTML compiler
```

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.8 to 3.11 installed.
- A Google Gemini API Key. (You can acquire one from [Google AI Studio](https://aistudio.google.com/)).

### Steps

1. **Clone or Navigate to the Directory**:
   ```bash
   cd DataMind_AI
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**:
   Create a `.env` file at the root of the project:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

6. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   The dashboard will launch in your default web browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions

### 1. Deploy on Streamlit Community Cloud (Recommended)
1. Commit the codebase to a public GitHub repository. (Exclude `.env` via `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**, select your repository, branch, and set the entrypoint file to `app.py`.
4. In the app settings under **Secrets**, add your Gemini API Key:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   ```
5. Click **Deploy!**

### 2. Deploy on Render (Dockerless Web Service)
1. Add a `build.sh` script or use Render's standard python service settings.
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Set Environment Variables in Render dashboard:
   - Key: `GEMINI_API_KEY`
   - Value: `your_gemini_api_key`

---

## 🚀 Future Roadmap & Scalability

- **Multi-File Datasets**: Allow uploading multiple datasets and joining them inside the chatbot execution context.
- **Advanced Predictive ML**: Integrate light scikit-learn models (regression, classification, time-series forecasting) that run automatically alongside pandas analysis.
- **Enhanced Code Sanitization**: Execute chatbot code within isolated webassembly / Docker sandboxes for maximum backend security in multi-tenant environments.
