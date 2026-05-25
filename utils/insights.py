import os
import google.generativeai as genai
from typing import Dict, Any, Optional

def get_best_model_name(api_key: str) -> str:
    """
    Queries the Gemini API for available models and returns the highest priority model.
    """
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name.replace("models/", "") for m in genai.list_models()]
        priority_list = [
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-pro"
        ]
        for model in priority_list:
            if model in available_models:
                return model
    except Exception:
        pass
    return "gemini-2.5-flash"  # default fallback

def generate_business_insights(
    dataset_summary: str,
    api_key: Optional[str] = None
) -> str:
    """
    Sends the dataset summary to Gemini and returns AI-generated business insights.
    """
    # 1. Resolve API key
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        return (
            "### ⚠️ Gemini API Key Missing\n"
            "Please configure your `GEMINI_API_KEY` in the sidebar or in the `.env` file to unlock AI Business Insights."
        )

    # 2. Configure GenAI and retrieve best model
    try:
        genai.configure(api_key=api_key)
        best_model = get_best_model_name(api_key)
        model = genai.GenerativeModel(best_model)
    except Exception as e:
        return f"### ❌ Error Configuring Gemini Client\nDetail: `{str(e)}`"

    # 3. Build the prompt
    system_instruction = (
        "You are an expert Chief Data Scientist, Business Analyst, and UI/UX Designer. "
        "Your goal is to inspect the provided metadata, statistics, and correlations of a dataset and produce "
        "a highly professional, executive-level business intelligence report. "
        "Do not use generic placeholders. Focus on uncovering high-value observations, actionable business recommendations, "
        "and potential anomalies. Format your output in clean, beautiful Markdown using standard headings (##, ###), "
        "bold text, bullet points, and code styling."
    )

    prompt = f"""
{system_instruction}

Here is the analytical summary of the dataset:
{dataset_summary}

Based on this data summary, please generate a detailed dashboard report structured exactly as follows:

## 📊 EXECUTIVE SUMMARY
A concise, high-level summary (2-3 sentences) of what this dataset represents, its overall quality, and the macro-level story it tells.

## 💡 KEY BUSINESS INSIGHTS
*Generate 3 to 5 highly specific, data-backed findings. Do not say generalities; reference specific column names, averages, maximums, or categories where relevant. Keep them punchy and clear.*

## 📈 TREND & CORRELATION ANALYSIS
*Identify patterns, temporal behaviors, or notable correlations between variables. If the correlation matrix indicates strong relationships, analyze why that might occur and what it implies for operations.*

## 🔍 ANOMALIES, RISKS & OUTLIERS
*Point out potential data quality risks, high numbers of nulls, unexpected outliers, or anomalous spikes/dips. Warn the business of potential pitfalls based on these patterns.*

## 🎯 STRATEGIC RECOMMENDATIONS
*Provide 3 to 4 actionable, concrete, and strategic decisions that business leaders should make based on these insights. Each recommendation should solve a problem or capitalize on an opportunity identified in the insights.*
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return (
            f"### ❌ Gemini API Execution Failed\n"
            f"There was an error communicating with the Gemini API. Please verify your API Key and network connection.\n\n"
            f"**Error Details:**\n`{str(e)}`"
        )
