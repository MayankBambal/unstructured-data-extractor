import re
from bs4 import BeautifulSoup

def text_table_to_markdown(table_text):
    """
    Convert a table (as plain text) into a Markdown table.
    Splits the text into lines, ignores separator lines (e.g., lines
    consisting only of dashes or equals), splits each row into columns
    using two or more whitespace characters, and builds a Markdown table
    with the first row as the header.
    """
    lines = table_text.splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        # Skip empty lines or lines that appear to be visual separators
        if not line or re.match(r'^[-=]+$', line):
            continue
        # Split the line on two or more whitespace characters
        cols = re.split(r'\s{2,}', line)
        if cols:
            rows.append(cols)
    
    if not rows:
        return ""
    
    # Build the Markdown table with the first row as the header
    md_lines = []
    header = rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)

def convert_tables_in_file_to_markdown(file_path):
    """
    Reads an HTML file from file_path, extracts all <table> elements, converts each to a
    Markdown table (ignoring tables that result in empty markdown), and returns a list
    of tuples containing (table_index, markdown_table).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    md_tables = []
    for idx, table in enumerate(tables, start=1):
        # Extract plain text from the table with newline separators to preserve rows
        table_text = table.get_text(separator='\n')
        md = text_table_to_markdown(table_text)
        if md:  # Only add non-empty markdown tables
            md_tables.append((idx, md))
    return md_tables 