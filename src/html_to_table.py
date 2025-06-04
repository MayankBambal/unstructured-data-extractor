import re
from bs4 import BeautifulSoup


def table_to_markdown(table):
    """
    Convert a BeautifulSoup table element to a Markdown table.
    Extracts text from all <th> and <td> cells (stripping any HTML/CSS),
    uses the first row as the header, and builds the Markdown table.
    """
    rows = []
    for tr in table.find_all('tr'):
        row = []
        for cell in tr.find_all(['th', 'td']):
            text = cell.get_text(separator=' ', strip=True)
            row.append(text)
        if row:
            rows.append(row)
    
    if not rows:
        return ""
    
    md_lines = []
    header = rows[0]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)

def convert_file(file_path):
    """
    Read an HTML file from file_path, extract all <table> elements,
    and convert each qualifying table into a Markdown table.
    A table qualifies if:
        - It contains at least one digit.
        - It has more than 5 rows.
        - (Previously: if the table had at least 3 rows, the third row with exactly 2 cells was skipped.
            This condition has been removed as per the request.)
    Returns a list of tuples: (table_index, markdown_table).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    md_tables = []
    
    for idx, table in enumerate(tables, start=1):
        # Filter: Only process tables that contain at least one digit.
        if not re.search(r'\d', table.get_text()):
            continue
        
        # Filter: Only process tables with more than 5 rows.
        all_rows = table.find_all('tr')
        if len(all_rows) <= 5:
            continue
        
        md = table_to_markdown(table)
        if md:
            md_tables.append((idx, md))
    
    return md_tables 