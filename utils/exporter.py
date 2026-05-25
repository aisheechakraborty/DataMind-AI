import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Converts a DataFrame to CSV bytes for Streamlit's download button.
    """
    return df.to_csv(index=False).encode('utf-8')

def figure_to_html_bytes(fig: go.Figure) -> bytes:
    """
    Converts a Plotly figure to standalone interactive HTML bytes.
    """
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    return html_str.encode('utf-8')

def markdown_to_html_report(insights_markdown: str, dataset_name: str, row_count: int, col_count: int) -> bytes:
    """
    Compiles AI insights Markdown into a beautiful, standalone, SaaS-themed HTML report.
    """
    # Simple markdown parser to convert basic elements to HTML for the exported report
    html_content = insights_markdown
    
    # Replace headings
    # Convert ### first, then ##, then # to avoid partial replacements
    import re
    html_content = re.sub(r'^###\s+(.*?)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^##\s+(.*?)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^#\s+(.*?)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
    
    # Replace bolding
    html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
    
    # Replace lists
    # Note: simple bullet conversion
    html_content = re.sub(r'^\*\s+(.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^-\s+(.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    
    # Group consecutive list items into <ul> blocks
    # We can wrap list segments
    html_content = re.sub(r'(<li>.*?</li>)+', r'<ul>\g<0></ul>', html_content, flags=re.DOTALL)
    # Fix nested <ul> from DOTALL matching too broadly
    html_content = html_content.replace('</ul><ul>', '')

    # Convert code backticks
    html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
    
    # Replace newlines with <br> inside paragraphs (excluding tag boundaries)
    # To keep it simple, we wrap sections in divs and paragraphs
    paragraphs = html_content.split('\n\n')
    formatted_paras = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith('<h') or para.startswith('<ul') or para.startswith('<li'):
            formatted_paras.append(para)
        else:
            formatted_paras.append(f"<p>{para.replace(chr(10), '<br>')}</p>")
            
    body_content = "\n".join(formatted_paras)

    # Combine into a premium themed template
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataMind AI Insights - {dataset_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0B0B0F;
            --card-bg: #13131A;
            --text-main: #FFFFFF;
            --text-muted: #A0A0AB;
            --accent-red: #E50914;
            --border-color: #23232C;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        
        .container {{
            max-width: 850px;
            width: 100%;
        }}
        
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 40px;
        }}
        
        .logo {{
            font-family: 'Outfit', sans-serif;
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: var(--text-main);
        }}
        
        .logo span {{
            color: var(--accent-red);
        }}
        
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .meta-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 15px;
            border-radius: 12px;
        }}
        
        .meta-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 5px;
        }}
        
        .meta-value {{
            font-size: 18px;
            font-weight: 700;
        }}
        
        h1, h2, h3 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-top: 40px;
            color: var(--text-main);
        }}
        
        h1 {{
            font-size: 36px;
        }}
        
        h2 {{
            font-size: 24px;
            border-left: 4px solid var(--accent-red);
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        
        h3 {{
            font-size: 18px;
            color: #F4F4F5;
        }}
        
        p {{
            color: #D4D4D8;
            font-size: 15px;
            line-height: 1.7;
            margin-bottom: 20px;
        }}
        
        ul {{
            padding-left: 20px;
            margin-bottom: 25px;
        }}
        
        li {{
            color: #D4D4D8;
            font-size: 15px;
            line-height: 1.7;
            margin-bottom: 10px;
        }}
        
        code {{
            background-color: #1E1E26;
            color: #FF7961;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 14px;
        }}
        
        .footer {{
            margin-top: 60px;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">DATAMIND<span>AI</span></div>
            <div class="meta-grid">
                <div class="meta-card">
                    <div class="meta-label">Dataset Analyzed</div>
                    <div class="meta-value" style="color: var(--accent-red);">{dataset_name}</div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Rows Count</div>
                    <div class="meta-value">{row_count:,}</div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Columns Count</div>
                    <div class="meta-value">{col_count}</div>
                </div>
            </div>
        </div>
        
        <div class="content">
            {body_content}
        </div>
        
        <div class="footer">
            Generated automatically by DataMind AI • CSV Analytics Agent &copy; 2026. All rights reserved.
        </div>
    </div>
</body>
</html>
"""
    return html_template.encode('utf-8')
