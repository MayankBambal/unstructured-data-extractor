# Usage Guide

## Quick Start

### 1. Setup Environment
```bash
# Set your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Basic Extraction
```bash
# Run basic extraction for NVDA (default company)
python src/main.py
```

### 3. Enhanced Extraction
```bash
# Run enhanced extraction with probability tracking
python src/main_enhanced.py
```

## Customizing the Extraction

### Change Target Company
Edit the ticker in `main.py` or `main_enhanced.py`:
```python
ticker = "AAPL"  # Change from "NVDA" to any ticker
```

### Change Query
Modify the extraction query:
```python
query = "Consolidated Statements of Operations"  # Instead of balance sheets
```

### Adjust Processing Years
```python
cutoff = 3  # Process last 3 years instead of 1
```

## Output Files

### Basic Extraction
- **Location**: `data/final/`
- **Files**: 
  - `{TICKER}_extracted.csv`: Main extraction results
  - `downloaded_forms.csv`: Filing metadata

### Enhanced Extraction
- **Location**: `data/enhanced_final/`
- **Files**:
  - `{TICKER}_call1_results.csv`: First extraction pass
  - `{TICKER}_call2_results.csv`: Second extraction pass
  - `enhanced_extraction_summary.csv`: Combined results with statistics

### Logs
- **Location**: `logs/`
- **Files**:
  - `main_pipeline.log`: Basic pipeline logs
  - `enhanced_pipeline.log`: Enhanced pipeline logs
  - `enhanced_extraction.log`: Detailed extraction logs

## Understanding the Output

### CSV Columns
- **Statement**: Type of financial statement (e.g., "Balance Sheet")
- **Item**: Line item name (e.g., "Total Assets")
- **Year**: Reporting year
- **Value**: Numerical value
- **Unit**: Unit of measurement (e.g., "thousands", "millions")
- **Ticker**: Company ticker symbol
- **File_year**: Year of the source filing

### Enhanced Extraction Additional Columns
- **model_probability**: AI confidence score (0.0 to 1.0)
- **confidence_level**: Classification (HIGH/MEDIUM/LOW)
- **extraction_timestamp**: When the extraction was performed

## Common Use Cases

### Extract Balance Sheet Data
```python
query = "Consolidated Balance Sheets"
```

### Extract Income Statement Data
```python
query = "Consolidated Statements of Operations"
```

### Extract Cash Flow Data
```python
query = "Consolidated Statements of Cash Flows"
```

### Extract Multiple Years
```python
cutoff = 5  # Last 5 years
```

## Troubleshooting

### No API Key Error
Make sure your `.env` file contains a valid OpenAI API key:
```bash
cat .env
# Should show: OPENAI_API_KEY=your_key_here
```

### No Filings Found
- Check if the ticker symbol is correct
- Verify the company has filed 10-K forms
- Check internet connection for SEC downloads

### Empty Extraction Results
- The query might not match content in the filing
- Try broader queries like "Balance Sheet" instead of "Consolidated Balance Sheets"
- Check the logs for extraction errors

### Low Confidence Scores
- Review the extracted data manually
- Consider adjusting the confidence threshold
- Use the dual-call enhanced extraction for better accuracy

## Advanced Configuration

### Custom Confidence Threshold
```python
# In enhanced extractor calls
confidence_threshold = 0.8  # Higher threshold = more conservative
```

### Custom OpenAI Model
Edit the extractor initialization:
```python
extractor = EnhancedFinancialExtractor(model="gpt-4")  # Use GPT-4 instead of GPT-4o-mini
```

### Batch Processing Multiple Companies
Create a script that loops through multiple tickers:
```python
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
for ticker in tickers:
    # Run extraction for each ticker
    pass
``` 