# SEC Financial Data Extractor

A Python-based tool for automatically extracting and processing financial data from SEC 10-K filings using OpenAI's GPT models. The system downloads SEC filings, cleans the data, and extracts structured financial information with confidence tracking.

## Features

- **Automated SEC Filing Download**: Downloads 10-K filings directly from SEC EDGAR database
- **AI-Powered Data Extraction**: Uses OpenAI GPT models for intelligent financial data extraction
- **Dual Extraction Methods**: Basic and enhanced extraction with probability tracking
- **Data Cleaning**: Preprocesses HTML filings for optimal extraction
- **Comprehensive Logging**: Detailed logging and monitoring of extraction processes
- **Structured Output**: Exports data to CSV with metadata and confidence scores

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Internet connection for SEC data downloading

### Setup
1. **Clone the repository**
```bash
git clone <repository-url>
cd unstructured-data-extractor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

5. **Verify installation**
```bash
python src/test_imports.py
```

## Usage

### Basic Extraction
```bash
python src/main.py
```

### Enhanced Extraction (with probability tracking)
```bash
python src/main_enhanced.py
```

### Configuration
The system is configured to process the most recent 10-K filing by default. Key settings:

- **Ticker**: Target company (default: "NVDA")
- **Form Type**: SEC form type (default: "10-K")
- **Cutoff**: Number of years to process (default: 1)
- **Query**: Extraction target (default: "Consolidated Balance Sheets")

## Project Structure

```
src/
├── main.py                    # Basic extraction pipeline
├── main_enhanced.py           # Enhanced pipeline with probability tracking
├── downloader.py              # SEC filing downloader
├── preprocessor.py            # HTML cleaning and preprocessing
├── extractor.py               # Basic OpenAI-powered extraction
├── enhanced_extractor.py      # Enhanced extraction with confidence metrics
├── html_to_table.py           # HTML table to markdown converter
├── plain_to_table.py          # Plain text table to markdown converter
└── test_imports.py            # Import verification script

data/
├── raw/                       # Downloaded SEC filings
├── processed/                 # Cleaned text files
├── final/                     # Basic extraction results
└── enhanced_final/            # Enhanced extraction results

logs/                          # Application logs
extracted/                     # Organized output files
```

## How It Works

1. **Download**: Fetches 10-K filings from SEC EDGAR database
2. **Clean**: Removes HTML/CSS and converts tables to markdown
3. **Extract**: Uses OpenAI GPT models to extract financial data
4. **Process**: Structures data into pandas DataFrames
5. **Export**: Saves results to CSV with metadata

## Core Components

### `downloader.py`
Downloads SEC filings and organizes them into structured directories.

### `preprocessor.py`
Cleans HTML content and converts tables to markdown format for better AI processing.

### `extractor.py` / `enhanced_extractor.py`
Core extraction engines that use OpenAI's API to extract structured financial data from cleaned filings.

### Main Scripts
- `main.py`: Basic extraction workflow
- `main_enhanced.py`: Enhanced workflow with probability tracking and confidence assessment

## Output Format

The system generates CSV files with the following structure:
- **Statement**: Financial statement type
- **Item**: Line item description
- **Year**: Reporting year
- **Value**: Numerical value
- **Unit**: Unit of measurement
- **Ticker**: Company ticker symbol
- **File_year**: Source filing year

Enhanced extraction also includes:
- **model_probability**: AI confidence score
- **confidence_level**: HIGH/MEDIUM/LOW classification

## Development

### Running Tests
```bash
python src/test_imports.py
```

### Adding New Extractors
Extend the base classes in `extractor.py` or `enhanced_extractor.py` to add new extraction capabilities.

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request 