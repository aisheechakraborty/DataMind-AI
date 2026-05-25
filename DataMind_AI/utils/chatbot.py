import os
import io
import sys
import traceback
import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import Dict, Any, List, Tuple, Optional

def execute_pandas_code(code_str: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Safely executes generated pandas code on the DataFrame, capturing printed output and 'result' variable.
    """
    # Extract code between triple backticks if present
    code = code_str
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
        
    code = code.strip()

    # Security check: intercept forbidden modules
    forbidden = ["os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "builtins.eval", "getattr", "setattr"]
    for keyword in forbidden:
        if keyword in code:
            return {
                "success": False,
                "error": f"Security restriction: Use of '{keyword}' is blocked in queries.",
                "code": code
            }

    # Redirect stdout
    stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout

    # Local context with pandas and numpy loaded, plus the user's DataFrame
    local_vars = {
        "df": df,
        "pd": pd,
        "np": np
    }

    try:
        # Execute the code block
        exec(code, {}, local_vars)
        sys.stdout = old_stdout
        
        printed = stdout.getvalue().strip()
        result = local_vars.get("result", None)

        if result is None:
            if printed:
                result = printed
            else:
                result = "Execution succeeded. No output was printed or saved to 'result'."

        return {
            "success": True,
            "result": result,
            "code": code,
            "printed": printed
        }
    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": tb,
            "code": code
        }

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

def chat_with_csv(
    df: pd.DataFrame,
    query: str,
    chat_history: List[Tuple[str, str]] = [],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates the LLM code-generation, execution, self-correction, and final response synthesis.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "answer": "⚠️ Gemini API Key missing. Please set the API key in the sidebar to use Chat.",
            "code": "",
            "success": False
        }

    try:
        genai.configure(api_key=api_key)
        best_model = get_best_model_name(api_key)
        model = genai.GenerativeModel(best_model)
    except Exception as e:
        return {
            "answer": f"❌ Error configuring Gemini API client: {str(e)}",
            "code": "",
            "success": False
        }

    # 1. Format metadata of df to guide Gemini code generation
    schema_info = []
    for col in df.columns:
        schema_info.append(f"- Column '{col}': dtype is {df[col].dtype}, has {df[col].nunique()} unique values. Examples: {list(df[col].dropna().head(3).tolist())}")
    schema_summary = "\n".join(schema_info)

    history_str = ""
    if chat_history:
        history_str = "Chat history:\n" + "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in chat_history]) + "\n\n"

    # Code Generator Prompt
    generator_prompt = f"""
You are a Python Data Analyst bot. Write a block of python code using pandas and numpy to answer the user's question about the dataframe `df`.
{history_str}
DataFrame `df` Schema & Samples:
{schema_summary}

Current User Question: "{query}"

Rules for writing code:
1. The DataFrame is loaded as `df`.
2. Do not define or load `df` yourself.
3. Compute the answer and save it in a variable called `result`. E.g., `result = df['Sales'].sum()`.
4. If the question asks for a list or table, create a pandas DataFrame/Series as the `result` or print it.
5. Keep your operations simple and efficient. Use accurate column names: {list(df.columns)}.
6. Do not import `os`, `sys`, `subprocess`, `shutil`, `socket`.
7. Output ONLY valid python code inside a single ```python and ``` code block. Do not add markdown descriptions before or after the code block.

Generate Python code now:
"""

    code_output = ""
    execution_result = {}
    retries = 2

    # Run Loop for self-correction
    for attempt in range(retries + 1):
        try:
            if attempt == 0:
                response = model.generate_content(generator_prompt)
                code_output = response.text.strip()
            else:
                # Feedback loop on error
                correction_prompt = f"""
The Python code you previously generated failed during execution on `df`.

Here is the code you generated:
```python
{execution_result.get('code', '')}
```

Here is the error traceback:
```
{execution_result.get('traceback', '')}
```

Please analyze the error and the dataframe columns: {list(df.columns)}.
Rewrite the code block to fix the error. Make sure to save the final output to `result` or print it.
Generate ONLY the corrected code inside a ```python and ``` block.
"""
                response = model.generate_content(correction_prompt)
                code_output = response.text.strip()
            
            # Execute generated code
            execution_result = execute_pandas_code(code_output, df)
            
            if execution_result["success"]:
                break  # Break out if execution is successful
                
        except Exception as e:
            execution_result = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "code": code_output
            }

    # If it failed all attempts, return error details to user
    if not execution_result.get("success", False):
        return {
            "answer": f"❌ I wrote Python code to analyze the CSV, but it encountered an error: `{execution_result.get('error')}`. Please try rephrasing your question.",
            "code": execution_result.get("code", code_output),
            "success": False
        }

    # 2. Synthesize final conversational answer
    res_val = execution_result["result"]
    # Truncate result if it is a massive string to avoid context overflows
    if isinstance(res_val, str) and len(res_val) > 4000:
        res_val = res_val[:4000] + "... [Truncated for brevity]"
    elif isinstance(res_val, pd.DataFrame):
        res_val = res_val.head(20).to_string()
    elif isinstance(res_val, pd.Series):
        res_val = res_val.head(20).to_string()

    synthesis_prompt = f"""
You are DataMind AI, a premium CSV analytics agent.
The user asked: "{query}"

We executed the following pandas code on their dataset:
```python
{execution_result['code']}
```

The execution output / result was:
```
{res_val}
```

Using this result, formulate a clear, polished, and direct answer to the user's question.
If the result contains numbers, explain them nicely (e.g. round floats, format as currency if they look like sales/revenue numbers, add commas for readability).
If the result is tabular, you can display it as a beautiful markdown table or lists.
Write the final response in markdown format. Keep it concise, professional, and friendly. Do not mention that you generated Python code under the hood unless they specifically ask how it was calculated.
"""

    try:
        final_response = model.generate_content(synthesis_prompt)
        return {
            "answer": final_response.text,
            "code": execution_result['code'],
            "success": True,
            "result_summary": str(res_val)
        }
    except Exception as e:
        return {
            "answer": f"I calculated the answer but failed to format the response.\nRaw Result:\n{str(res_val)}",
            "code": execution_result['code'],
            "success": True
        }
