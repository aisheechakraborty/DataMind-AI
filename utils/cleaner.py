import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    fill_numeric: str = "median",  # "mean", "median", "mode", "zero", "drop", "none"
    fill_categorical: str = "mode",  # "mode", "unknown", "drop", "none"
    clean_strings: bool = True,
    coerce_types: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans a pandas DataFrame and returns the cleaned DataFrame and a log of changes.
    """
    cleaned_df = df.copy()
    changes_log = {
        "initial_rows": len(df),
        "initial_cols": len(df.columns),
        "duplicates_removed": 0,
        "missing_filled": {},
        "errors_coerced": {},
        "final_rows": 0,
        "final_cols": 0
    }

    # 1. Remove duplicates
    if remove_duplicates:
        duplicate_count = cleaned_df.duplicated().sum()
        if duplicate_count > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
            changes_log["duplicates_removed"] = int(duplicate_count)

    # 2. String Cleaning
    if clean_strings:
        for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
            try:
                # Strip whitespace
                cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                # Replace empty strings with NaN
                cleaned_df[col] = cleaned_df[col].replace(r'^\s*$', np.nan, regex=True)
            except Exception:
                pass

    # 3. Automatic Type Coercion / Data Cleaning
    # Detect columns that are mostly numeric but read as object/string due to noise, e.g. currency signs, commas
    if coerce_types:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                # Skip if already identified as a datetime or if too many non-numeric characters
                # Let's check if we can parse it as a float after removing currency signs, spaces, percent signs, commas
                sample_vals = cleaned_df[col].dropna().head(100).astype(str)
                if len(sample_vals) > 0:
                    cleaned_samples = sample_vals.str.replace(r'[\$,%]', '', regex=True).str.replace(r'\s+', '', regex=True).str.replace(',', '', regex=False)
                    # Try to convert to numeric
                    try:
                        numeric_converted = pd.to_numeric(cleaned_samples, errors='coerce')
                        # If more than 80% of non-null values convert successfully to numbers, coerce the entire column
                        if (numeric_converted.notna().sum() / len(cleaned_samples)) > 0.8:
                            before_nas = cleaned_df[col].isna().sum()
                            coerced_col = pd.to_numeric(
                                cleaned_df[col].astype(str).str.replace(r'[\$,%]', '', regex=True).str.replace(r'\s+', '', regex=True).str.replace(',', '', regex=False),
                                errors='coerce'
                            )
                            after_nas = coerced_col.isna().sum()
                            new_nas_created = int(after_nas - before_nas)
                            if new_nas_created > 0:
                                changes_log["errors_coerced"][col] = new_nas_created
                            cleaned_df[col] = coerced_col
                    except Exception:
                        pass

    # 4. Handle Missing Values
    for col in cleaned_df.columns:
        null_count = cleaned_df[col].isna().sum()
        if null_count > 0:
            changes_log["missing_filled"][col] = {
                "null_count": int(null_count),
                "strategy": "none"
            }
            
            # Numeric columns (excluding booleans which pandas classifies as numeric)
            if pd.api.types.is_numeric_dtype(cleaned_df[col]) and not pd.api.types.is_bool_dtype(cleaned_df[col]):
                try:
                    if fill_numeric == "mean":
                        val = cleaned_df[col].mean()
                        cleaned_df[col] = cleaned_df[col].fillna(val)
                        changes_log["missing_filled"][col]["strategy"] = f"filled with mean ({val:.2f})"
                    elif fill_numeric == "median":
                        val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(val)
                        changes_log["missing_filled"][col]["strategy"] = f"filled with median ({val:.2f})"
                    elif fill_numeric == "mode":
                        mode_val = cleaned_df[col].mode()
                        val = mode_val.iloc[0] if not mode_val.empty else 0
                        cleaned_df[col] = cleaned_df[col].fillna(val)
                        changes_log["missing_filled"][col]["strategy"] = f"filled with mode ({val})"
                    elif fill_numeric == "zero":
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                        changes_log["missing_filled"][col]["strategy"] = "filled with 0"
                    elif fill_numeric == "drop":
                        cleaned_df = cleaned_df.dropna(subset=[col])
                        changes_log["missing_filled"][col]["strategy"] = "rows dropped"
                except Exception as e:
                    changes_log["missing_filled"][col]["strategy"] = f"imputation failed ({str(e)})"
            
            # Categorical, Boolean, Datetime columns
            else:
                try:
                    if fill_categorical == "mode":
                        mode_val = cleaned_df[col].mode()
                        if not mode_val.empty:
                            val = mode_val.iloc[0]
                            cleaned_df[col] = cleaned_df[col].fillna(val)
                            changes_log["missing_filled"][col]["strategy"] = f"filled with mode ('{val}')"
                        else:
                            # Fallback if no mode available
                            if not pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                                cleaned_df[col] = cleaned_df[col].astype(object).fillna("Unknown")
                                changes_log["missing_filled"][col]["strategy"] = "filled with 'Unknown'"
                    elif fill_categorical == "unknown":
                        if pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                            # Do not fill datetime with string, leave as NaT or drop
                            pass
                        else:
                            cleaned_df[col] = cleaned_df[col].astype(object).fillna("Unknown")
                            changes_log["missing_filled"][col]["strategy"] = "filled with 'Unknown'"
                    elif fill_categorical == "drop":
                        cleaned_df = cleaned_df.dropna(subset=[col])
                        changes_log["missing_filled"][col]["strategy"] = "rows dropped"
                except Exception as e:
                    changes_log["missing_filled"][col]["strategy"] = f"imputation failed ({str(e)})"

    # Reset index and recount final dimensions
    cleaned_df = cleaned_df.reset_index(drop=True)
    changes_log["final_rows"] = len(cleaned_df)
    changes_log["final_cols"] = len(cleaned_df.columns)
    
    return cleaned_df, changes_log
