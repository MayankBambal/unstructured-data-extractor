# API Reference

## Core Classes

### `Config` (downloader.py)
Configuration class for SEC data downloading.

**Attributes:**
- `base_dir`: Base directory for data storage
- `data_dir`: Directory for raw data
- `staging_dir`: Directory for staging files

### `SECDataLoader` (downloader.py)
Downloads and manages SEC filing data.

**Methods:**
- `download_sec_filing(form_type, ticker)`: Download SEC filings
- `get_downloaded_forms_df()`: Get DataFrame of downloaded forms
- `rename_and_copy_filings()`: Organize downloaded files
- `list_downloaded_filings(ticker, form_type, cutoff)`: List available filings

### `DataCleaner` (preprocessor.py)
Cleans and preprocesses HTML filing data.

**Usage:**
```python
cleaned_text = DataCleaner(file_path)
```

### `FinancialDataExtractorOpenAI` (extractor.py)
Basic OpenAI-powered financial data extraction.

**Methods:**
- `extract_from_file(file_path, query)`: Extract data from file
- `process_and_flatten(json_response)`: Process JSON to DataFrame

**Configuration:**
- Model: GPT-4o-mini
- Temperature: 0.1
- Max tokens: 4000

### `EnhancedFinancialExtractor` (enhanced_extractor.py)
Enhanced extraction with probability tracking and confidence assessment.

**Methods:**
- `extract_from_file(file_path, query, confidence_threshold=0.7)`: Extract with confidence filtering
- `extract_from_file_both_calls(file_path, query, confidence_threshold=0.7)`: Dual extraction calls

**Features:**
- Probability tracking
- Confidence classification (HIGH/MEDIUM/LOW)
- Timeout handling
- JSON repair capabilities

## Utility Functions

### HTML/Table Processing
- `table_to_markdown(table)`: Convert BeautifulSoup table to markdown
- `convert_file(file_path)`: Extract tables from HTML files
- `text_table_to_markdown(table_text)`: Convert plain text tables to markdown

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI extraction

### Default Settings
- **Company**: NVDA (configurable in main scripts)
- **Form Type**: 10-K
- **Processing**: Single year (most recent)
- **Query**: "Consolidated Balance Sheets"
- **Confidence Threshold**: 0.7 (enhanced extractor)

## Output Schema

### Basic Extraction
```
Statement | Item | Year | Value | Unit | Ticker | File_year
```

### Enhanced Extraction (Additional Fields)
```
model_probability | confidence_level | extraction_timestamp
```

## Error Handling

All components include comprehensive error handling with logging:
- Network timeouts for SEC downloads
- OpenAI API errors and rate limiting
- JSON parsing errors with repair attempts
- File I/O errors

## Logging

Logs are written to the `logs/` directory:
- `main_pipeline.log`: Basic extraction pipeline
- `enhanced_pipeline.log`: Enhanced extraction pipeline
- `enhanced_extraction.log`: Detailed extraction logs 