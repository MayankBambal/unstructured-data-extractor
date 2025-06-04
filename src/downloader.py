import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import shutil

class Config:
    """Configuration settings for the SEC data extractor."""
    SEC_HEADERS = {'User-Agent': "dogesh@dogdash.com"}
    SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    DOWNLOADER_NAME = "Dogdash"
    DOWNLOADER_EMAIL = "dogesh@dogdash.com"
    OUTPUT_DIR = "data/staging"  # Updated output directory

class SECDataLoader:
    """Class for loading and managing SEC filing data."""
    def __init__(self, config: Config):
        self.config = config
        self.company_data = None
        # Pass output_dir to the Downloader (if supported by the library)
        self.downloader = Downloader(
            self.config.DOWNLOADER_NAME,
            self.config.DOWNLOADER_EMAIL,
            self.config.OUTPUT_DIR  # Ensures filings are saved under data/staging
        )

    def get_company_tickers(self) -> pd.DataFrame:
        """Fetch company tickers data from the SEC and return as a DataFrame."""
        response = requests.get(
            self.config.SEC_COMPANY_TICKERS_URL,
            headers=self.config.SEC_HEADERS
        )
        if response.status_code != 200:
            raise Exception(f"Failed to fetch company tickers: {response.status_code}")
        company_data = pd.DataFrame.from_dict(response.json(), orient='index')
        company_data['cik_str'] = company_data['cik_str'].astype(str).str.zfill(10)
        self.company_data = company_data
        return company_data

    def download_sec_filing(self, form_type: str, ticker_or_cik: str) -> str:
        """
        Download SEC filings for a specific form type and company.
        Returns the directory path where filings are downloaded.
        """
        print(f"Downloading {form_type} filings for {ticker_or_cik}...")
        # Download using the sec_edgar_downloader
        self.downloader.get(form_type, ticker_or_cik)
        # Note: the downloader library may create an intermediate folder (e.g., "sec-edgar-filings")
        # inside OUTPUT_DIR. Return the base directory for the specific ticker and form type.
        base_path = os.path.join(self.config.OUTPUT_DIR, ticker_or_cik.upper(), form_type)
        if not os.path.exists(base_path):
            # In case the downloader creates an extra folder (like "sec-edgar-filings"), search for it.
            for folder in os.listdir(self.config.OUTPUT_DIR):
                potential = os.path.join(self.config.OUTPUT_DIR, folder, ticker_or_cik.upper(), form_type)
                if os.path.exists(potential):
                    base_path = potential
                    break
        return base_path

    def list_downloaded_filings(self, ticker_or_cik: str, form_type: str, cutoff_year: int) -> list:
        """
        List paths to renamed downloaded filings from the data/raw directory for the given ticker and form type,
        returning the most recent filings up to cutoff_year count.
        
        Expected file name format in data/raw: <TICKER>-10k-<year>.txt
        (e.g., AAPL-10k-20.txt)
        """
        raw_dir = os.path.join("data", "raw")
        if not os.path.exists(raw_dir):
            return []
        
        # Convert form_type to token format, e.g., "10-K" -> "10k"
        token = form_type.replace("-", "").lower()
        ticker_upper = ticker_or_cik.upper()
        matched_files = []
        
        for filename in os.listdir(raw_dir):
            if filename.endswith(".txt") and filename.startswith(f"{ticker_upper}-{token}-"):
                parts = filename.split('-')
                if len(parts) < 3:
                    continue  # Skip files that don't match the expected naming convention.
                # Extract the year from the third part (removing the .txt extension)
                year_str = parts[2].replace(".txt", "")
                try:
                    year_int = int(year_str)
                    matched_files.append((year_int, os.path.join(raw_dir, filename)))
                except ValueError:
                    # Skip files where the year is not a valid integer.
                    continue
        
        # Sort by year in descending order (most recent first) and return up to cutoff_year count
        matched_files.sort(key=lambda x: x[0], reverse=True)
        return [file_path for _, file_path in matched_files[:cutoff_year]]

    
    def get_downloaded_forms_df(self) -> pd.DataFrame:
        """
        Scan the download directory and build a DataFrame of downloaded forms.
        The folder structure is expected to be:
            <OUTPUT_DIR>/(optional subfolder e.g., "sec-edgar-filings")/<TICKER>/<FORM_TYPE>/<cik>-<year>-<file number>
        Returns a DataFrame with columns: ticker, form_type, cik, year, filer_number.
        """
        # Determine the base directory where the downloaded forms are stored.
        base_dir = self.config.OUTPUT_DIR
        # Check if there's an extra folder (e.g., "sec-edgar-filings") in the OUTPUT_DIR.
        possible_subfolder = os.path.join(base_dir, "sec-edgar-filings")
        if os.path.isdir(possible_subfolder):
            base_dir = possible_subfolder
        
        rows = []
        if not os.path.exists(base_dir):
            return pd.DataFrame(columns=["ticker", "form_type", "cik", "year", "filer_number"])
        
        # Loop over each ticker directory.
        for ticker in os.listdir(base_dir):
            ticker_path = os.path.join(base_dir, ticker)
            if os.path.isdir(ticker_path):
                # Loop over each form type directory.
                for form_type in os.listdir(ticker_path):
                    form_type_path = os.path.join(ticker_path, form_type)
                    if os.path.isdir(form_type_path):
                        # Each folder here is expected to be named like: <cik>-<year>-<file number>
                        for form_folder in os.listdir(form_type_path):
                            form_folder_path = os.path.join(form_type_path, form_folder)
                            if os.path.isdir(form_folder_path):
                                parts = form_folder.split('-')
                                if len(parts) >= 3:
                                    cik = parts[0]
                                    year = parts[1]
                                    filer_number = parts[2]
                                    rows.append({
                                        "ticker": ticker,
                                        "form_type": form_type,
                                        "cik": cik,
                                        "year": year,
                                        "filer_number": filer_number
                                    })
        df = pd.DataFrame(rows, columns=["ticker", "form_type", "cik", "year", "filer_number"])
        return df
    
    def rename_and_copy_filings(self):
        """
        Traverse each filing folder under the download directory, rename the full-submission.txt file to
        <ticker>-10k-<year>.txt (based on the parent folder name format <cik>-<year>-<file number>),
        and copy the renamed file into the data/raw directory.
        """
        raw_dir = os.path.join("data", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Determine the base directory; if there's an extra folder (e.g., "sec-edgar-filings"), use it.
        base_dir = self.config.OUTPUT_DIR
        possible_subfolder = os.path.join(base_dir, "sec-edgar-filings")
        if os.path.isdir(possible_subfolder):
            base_dir = possible_subfolder

        # Traverse the directory structure: <base_dir>/<TICKER>/<FORM_TYPE>/<folder>/full-submission.txt
        for ticker in os.listdir(base_dir):
            ticker_path = os.path.join(base_dir, ticker)
            if os.path.isdir(ticker_path):
                for form_type in os.listdir(ticker_path):
                    form_type_path = os.path.join(ticker_path, form_type)
                    if os.path.isdir(form_type_path):
                        for folder in os.listdir(form_type_path):
                            folder_path = os.path.join(form_type_path, folder)
                            if os.path.isdir(folder_path):
                                # Construct the expected full submission file path.
                                submission_file = os.path.join(folder_path, "full-submission.txt")
                                if os.path.exists(submission_file):
                                    parts = folder.split('-')
                                    if len(parts) >= 3:
                                        # Extract the year from the folder name (second part).
                                        year = parts[1]
                                        # Create the new file name as <ticker>-10k-<year>.txt.
                                        new_filename = f"{ticker}-10k-{year}.txt"
                                        destination = os.path.join(raw_dir, new_filename)
                                        shutil.copy2(submission_file, destination)
                                        print(f"Copied {submission_file} to {destination}")
                                    else:
                                        print(f"Skipping folder {folder} as it does not match expected naming convention.")
