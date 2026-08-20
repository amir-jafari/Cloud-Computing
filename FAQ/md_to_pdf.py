#!/usr/bin/env python3
"""
md_to_pdf.py — Convert a Markdown file to a styled PDF.

Requirements:
    pip install markdown --break-system-packages

    Plus one HTML-to-PDF renderer, tried in this order:
      1. wkhtmltopdf on PATH or in a default install dir
         (Debian/Ubuntu: apt-get install wkhtmltopdf)
      2. Headless Chrome / Chromium / Microsoft Edge
         (already present on most Windows and macOS machines)
      3. pip install weasyprint   (needs GTK libraries on Windows)
      4. pip install xhtml2pdf    (pure Python, weakest CSS support)

Usage:
    python3 md_to_pdf.py input.md output.pdf
"""

import sys
import shutil
import subprocess
import tempfile
import os
import markdown


CSS = """
@page { margin: 22mm 18mm; }
body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    color: #1f2430;
    font-size: 11pt;
    line-height: 1.5;
}
h1 {
    font-size: 22pt;
    color: #0f172a;
    border-bottom: 3px solid #2563eb;
    padding-bottom: 8px;
    margin-bottom: 4px;
}
h1 + p { color: #475569; font-size: 11pt; margin-top: 6px; }
h2 {
    font-size: 15pt;
    color: #1d4ed8;
    margin-top: 26px;
    margin-bottom: 8px;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
}
h3 { font-size: 12.5pt; color: #334155; margin-top: 14px; margin-bottom: 8px; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 4px 0; }
ul { margin: 4px 0 10px 0; padding-left: 22px; }
ul.task-list { list-style: none; padding-left: 4px; }
ul.task-list > li { margin-bottom: 7px; }
li { margin-bottom: 3px; }
li ul { margin-top: 4px; margin-bottom: 6px; }
input[type=checkbox] {
    margin-right: 8px;
    transform: scale(1.15);
    position: relative;
    top: 1px;
}
code {
    background: #f1f5f9;
    color: #b91c1c;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
}
pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 14px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 7.6pt;
    line-height: 1.35;
}
pre code { background: none; color: inherit; padding: 0; font-size: inherit; }
strong { color: #0f172a; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
th { background: #f1f5f9; }
"""


def _find_wkhtmltopdf() -> str | None:
    """Return a wkhtmltopdf executable, checking PATH then default install dirs."""
    found = shutil.which("wkhtmltopdf")
    if found:
        return found

    candidates = [
        r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
        r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
        "/usr/local/bin/wkhtmltopdf",
        "/opt/homebrew/bin/wkhtmltopdf",
    ]
    return next((c for c in candidates if os.path.isfile(c)), None)


def _find_chrome() -> str | None:
    """Return a Chrome/Chromium/Edge executable usable for headless printing."""
    for name in ("chrome", "chromium", "chromium-browser", "google-chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    return next((c for c in candidates if os.path.isfile(c)), None)


def html_to_pdf(html: str, html_path: str, pdf_path: str) -> None:
    """Render `html` (already written to `html_path`) with the first backend available."""
    pdf_path = os.path.abspath(pdf_path)

    wkhtmltopdf = _find_wkhtmltopdf()
    if wkhtmltopdf:
        subprocess.run(
            [
                wkhtmltopdf,
                "--enable-local-file-access",
                "--encoding",
                "utf-8",
                html_path,
                pdf_path,
            ],
            check=True,
        )
        return

    chrome = _find_chrome()
    if chrome:
        # Chrome refuses to overwrite nothing, but it also silently keeps a stale
        # file on failure, so clear the target first and verify afterwards.
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        with tempfile.TemporaryDirectory() as profile_dir:
            subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-pdf-header-footer",
                    f"--user-data-dir={profile_dir}",
                    f"--print-to-pdf={pdf_path}",
                    _file_url(html_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if os.path.isfile(pdf_path):
            return
        print("Headless browser produced no PDF, trying a Python renderer...")

    try:
        from weasyprint import HTML  # noqa: PLC0415

        HTML(string=html, base_url=os.path.dirname(os.path.abspath(html_path))).write_pdf(
            pdf_path
        )
        return
    except Exception:
        pass

    try:
        from xhtml2pdf import pisa  # noqa: PLC0415

        with open(pdf_path, "wb") as out:
            status = pisa.CreatePDF(html, dest=out, encoding="utf-8")
        if not status.err:
            return
    except ImportError:
        pass

    raise RuntimeError(
        "No HTML-to-PDF renderer found. Install one of:\n"
        "  - wkhtmltopdf  (https://wkhtmltopdf.org/downloads.html)\n"
        "  - Google Chrome or Microsoft Edge\n"
        "  - pip install weasyprint\n"
        "  - pip install xhtml2pdf"
    )


def _file_url(path: str) -> str:
    from pathlib import Path  # noqa: PLC0415

    return Path(os.path.abspath(path)).as_uri()


def convert(md_path: str, pdf_path: str) -> None:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "fenced_code", "tables", "sane_lists", "toc"],
        extension_configs={},
    )
    # python-markdown doesn't natively support GitHub-style "- [ ]" task
    # lists, so convert them to real checkboxes after the fact.
    html_body = html_body.replace("<li>[ ] ", '<li><input type="checkbox"> ')
    html_body = html_body.replace("<li>[x] ", '<li><input type="checkbox" checked> ')

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{os.path.basename(md_path)}</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp_html:
        tmp_html.write(full_html)
        tmp_html_path = tmp_html.name

    try:
        html_to_pdf(full_html, tmp_html_path, pdf_path)
    finally:
        os.remove(tmp_html_path)

    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 md_to_pdf.py input.md output.pdf")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
