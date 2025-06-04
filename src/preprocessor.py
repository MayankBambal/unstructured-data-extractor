from bs4 import BeautifulSoup
from html_to_table import convert_file
from plain_to_table import convert_tables_in_file_to_markdown


def DataCleaner(file_path):
    """
    Reads the HTML file and classifies it as either:
      - "HTML with CSS elements" if it contains any <style> tags, 
        <link> tags with rel="stylesheet", or any inline style attributes.
      - "HTML with only plain HTML tags" if none of these are found.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Check for <style> tags.
    has_style_tag = bool(soup.find('style'))
    
    # Check for external stylesheet links.
    has_link_stylesheet = bool(soup.find('link', rel=lambda x: x and 'stylesheet' in x.lower()))
    
    # Check for any inline style attributes.
    has_inline_style = any(tag.has_attr('style') for tag in soup.find_all())
    
    if has_style_tag or has_link_stylesheet or has_inline_style:
        tables = convert_file(file_path)
        return tables
    else:
        tables = convert_tables_in_file_to_markdown(file_path)
        return tables
