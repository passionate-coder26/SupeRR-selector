"""
Export SUBMISSION_RANKINGS_AND_REPORT.md to Publication-Quality PDF
with full LaTeX math rendering via MathJax v3 and print-fitted tables.
"""

import os
import re
import shutil
import subprocess
import sys
import markdown

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD_PATH = os.path.join(PROJECT_ROOT, "SUBMISSION_RANKINGS_AND_REPORT.md")
HTML_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "submission_report_print.html")
PDF_PATH = os.path.join(PROJECT_ROOT, "SUBMISSION_RANKINGS_AND_REPORT.pdf")
LOCAL_MATHJAX = os.path.join(PROJECT_ROOT, "assets", "mathjax", "tex-mml-chtml.js")


def find_browser_executable():
    """Locate Microsoft Edge or Chrome executable on Windows/OS."""
    candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def convert_markdown_with_math_protection(md_text):
    """
    Protects LaTeX equations ($...$ and $$...$$) from Markdown underscore/emphasis
    parsing, processes tables and fenced code, then restores the exact math expressions.
    """
    math_store = {}
    counter = [0]

    def save_display_math(match):
        token = f"@@@MATHDISPLAY{counter[0]}@@@"
        math_store[token] = match.group(0)
        counter[0] += 1
        return token

    def save_inline_math(match):
        token = f"@@@MATHINLINE{counter[0]}@@@"
        math_store[token] = match.group(0)
        counter[0] += 1
        return token

    # 1. Protect display math $$...$$
    text = re.sub(r"\$\$(.*?)\$\$", save_display_math, md_text, flags=re.DOTALL)

    # 2. Protect inline math $...$ (avoiding double-dollar and empty dollars)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", save_inline_math, text)

    # 3. Convert Markdown to HTML with tables & code extensions
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

    # 4. Restore math expressions
    for token, math_expr in math_store.items():
        html_body = html_body.replace(token, math_expr)

    return html_body


def build_complete_html(html_body):
    """Wraps body in an A4 print-optimized template with Rajasthan Royals branding and MathJax v3."""
    # Read local MathJax script if available for fast offline rendering
    local_mathjax_script = ""
    if os.path.exists(LOCAL_MATHJAX):
        with open(LOCAL_MATHJAX, "r", encoding="utf-8") as f:
            local_mathjax_script = f"<script>{f.read()}</script>"
    else:
        local_mathjax_script = '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Rajasthan Royals SupeRR Selector S2 — Official Methodology Report</title>
  
  <!-- MathJax Configuration -->
  <script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
      processEscapes: true
    }},
    options: {{
      renderActions: {{
        addMenu: []
      }}
    }},
    startup: {{
      pageReady: () => {{
        return MathJax.startup.defaultPageReady().then(() => {{
          document.body.classList.add('mathjax-typeset-complete');
          console.log('MathJax typesetting complete');
        }});
      }}
    }}
  }};
  </script>
  {local_mathjax_script}

  <style>
    @page {{
      size: A4 portrait;
      margin: 12mm 12mm 12mm 12mm;
      @bottom-right {{
        content: counter(page);
        font-size: 8pt;
        color: #64748B;
      }}
    }}

    *, *:before, *:after {{
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: #0F172A;
      background-color: #FFFFFF;
      line-height: 1.45;
      font-size: 9.5pt;
      margin: 0;
      padding: 0;
    }}

    h1 {{
      color: #070E22;
      font-size: 18pt;
      font-weight: 800;
      border-bottom: 3px solid #EA1A85;
      padding-bottom: 6px;
      margin-top: 0;
      margin-bottom: 10px;
    }}

    h2 {{
      color: #070E22;
      font-size: 13.5pt;
      font-weight: 700;
      border-bottom: 1.5px solid #EA1A85;
      padding-bottom: 4px;
      margin-top: 20px;
      margin-bottom: 8px;
      page-break-after: avoid;
    }}

    h3 {{
      color: #070E22;
      font-size: 11pt;
      font-weight: 700;
      margin-top: 14px;
      margin-bottom: 6px;
      page-break-after: avoid;
    }}

    h4 {{
      color: #070E22;
      font-size: 10pt;
      font-weight: 600;
      margin-top: 10px;
      margin-bottom: 4px;
      page-break-after: avoid;
    }}

    p {{
      margin: 6px 0;
    }}

    hr {{
      border: 0;
      border-top: 1px solid #E2E8F0;
      margin: 14px 0;
    }}

    /* Print-fitted Tables */
    table {{
      width: 100% !important;
      max-width: 100% !important;
      table-layout: auto !important;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 8.2pt;
      page-break-inside: avoid;
      word-wrap: break-word;
    }}

    th {{
      background-color: #070E22 !important;
      color: #FFFFFF !important;
      border: 1px solid #1E293B;
      border-bottom: 2px solid #EA1A85 !important;
      padding: 4px 5px;
      text-align: center;
      font-weight: 700;
      font-size: 8pt;
      vertical-align: middle;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    td {{
      border: 1px solid #CBD5E1;
      padding: 3.5px 5px;
      vertical-align: middle;
      font-size: 7.8pt;
      text-align: left;
    }}

    td:first-child {{
      text-align: center;
    }}

    tr:nth-child(even) td {{
      background-color: #F8FAFC !important;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    blockquote {{
      background-color: #F8FAFC;
      border-left: 3.5px solid #EA1A85;
      padding: 6px 12px;
      margin: 8px 0;
      font-size: 8.8pt;
      color: #334155;
      page-break-inside: avoid;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    code {{
      background-color: #F1F5F9;
      color: #EA1A85;
      padding: 1px 3.5px;
      border-radius: 3px;
      font-size: 8pt;
      font-family: Consolas, "Courier New", monospace;
    }}

    pre {{
      background-color: #0F172A;
      color: #F8FAFC;
      padding: 8px 12px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 8pt;
      page-break-inside: avoid;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    pre code {{
      background: none;
      color: inherit;
      padding: 0;
    }}

    ul, ol {{
      margin: 5px 0 8px 18px;
      padding: 0;
    }}

    li {{
      margin: 2px 0;
    }}

    /* MathJax Print Sizing */
    mjx-container {{
      page-break-inside: avoid;
      color: #0F172A;
    }}

    mjx-container[jax="CHTML"][display="true"] {{
      margin: 8px 0 !important;
      font-size: 105% !important;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    return html_template


def export_pdf():
    print("=" * 80)
    print("📄 EXPORTING PUBLICATION-GRADE PDF: SUBMISSION_RANKINGS_AND_REPORT.pdf")
    print("=" * 80)

    if not os.path.exists(MD_PATH):
        print(f"❌ Error: {MD_PATH} does not exist.")
        sys.exit(1)

    print(f"1. Reading Markdown report: {MD_PATH}...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    print("2. Protecting LaTeX equations and converting Markdown to HTML...")
    html_body = convert_markdown_with_math_protection(md_content)

    print("3. Wrapping HTML with print stylesheet and MathJax configuration...")
    full_html = build_complete_html(html_body)

    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"  ✓ Exported print HTML: {HTML_PATH} ({len(full_html):,} bytes)")

    browser_bin = find_browser_executable()
    if not browser_bin:
        print("❌ Error: Microsoft Edge or Chrome executable not found in PATH or standard directories.")
        sys.exit(1)

    print(f"4. Rendering PDF via Headless Browser: {browser_bin}...")
    # Convert file path to file URL
    html_file_url = "file:///" + HTML_PATH.replace("\\", "/")

    temp_profile = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "edge_pdf_profile")
    os.makedirs(temp_profile, exist_ok=True)

    cmd = [
        browser_bin,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        f"--user-data-dir={temp_profile}",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        f'--print-to-pdf={PDF_PATH}',
        html_file_url
    ]

    print(f"  Running command: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  ⚠️ Warning: Browser process returned code {result.returncode}")
        if result.stderr:
            print(f"  Browser stderr: {result.stderr}")

    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 0:
        size_kb = os.path.getsize(PDF_PATH) / 1024.0
        print(f"\n✅ SUCCESS! SUBMISSION_RANKINGS_AND_REPORT.pdf generated successfully!")
        print(f"   Path: {PDF_PATH}")
        print(f"   Size: {size_kb:.1f} KB")
        print("=" * 80)
        return True
    else:
        print(f"\n❌ Error: Failed to produce non-empty PDF at {PDF_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    export_pdf()
