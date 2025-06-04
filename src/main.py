from downloader import Config, SECDataLoader
from preprocessor import DataCleaner
from extractor import FinancialDataExtractorOpenAI
import os
import pandas as pd
import logging
import time
import sys

# Configure main pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("=== FINANCIAL DATA EXTRACTION PIPELINE STARTED ===")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize configuration and SECDataLoader
        logger.info("=== STEP 1: INITIALIZATION ===")
        config = Config()
        sec_loader = SECDataLoader(config)
        logger.info("SEC Data Loader initialized successfully")
        
        # Download 10-K filings for NVDA - Only one year
        form_type = "10-K"
        ticker = "NVDA"
        cutoff = 1  # Only process one year of reports
        logger.info(f"Target: {ticker} {form_type} filings (cutoff: {cutoff} - single year only)")
        
        # Step 1.1: Download SEC filings
        logger.info("Step 1.1: Downloading SEC filings...")
        download_start = time.time()
        download_dir = sec_loader.download_sec_filing(form_type, ticker)
        download_time = time.time() - download_start
        logger.info(f"Filings downloaded to: {download_dir} (took {download_time:.2f} seconds)")

        # Create a DataFrame from the folder names and save it.
        logger.info("Step 1.2: Creating downloaded forms DataFrame...")
        downloaded_forms_df = sec_loader.get_downloaded_forms_df()
        final_dir = os.path.join("data", "final")
        os.makedirs(final_dir, exist_ok=True)
        forms_csv_path = os.path.join(final_dir, "downloaded_forms.csv")
        downloaded_forms_df.to_csv(forms_csv_path, index=False)
        logger.info(f"Downloaded forms information saved to: {forms_csv_path}")
        logger.info(f"Downloaded forms shape: {downloaded_forms_df.shape}")
        
        # Step 1.3: Rename and copy filings
        logger.info("Step 1.3: Renaming and copying filings...")
        sec_loader.rename_and_copy_filings()
        logger.info("Filing rename and copy complete")
        
        # List downloaded filings (looking for full-submission.txt files)
        logger.info("Step 1.4: Listing downloaded filings...")
        filing_files = sec_loader.list_downloaded_filings(ticker, form_type, cutoff)
        if not filing_files:
            logger.error("No filings found in the downloaded directory.")
            return
        
        logger.info(f"Found {len(filing_files)} filing files:")
        for i, file_path in enumerate(filing_files, 1):
            logger.info(f"  {i}. {file_path}")

        # Initialize a list to collect DataFrames from each filing.
        final_dataframes = []

        # Instantiate the extractor once (if the API key/configuration is constant).
        logger.info("=== STEP 2: EXTRACTOR INITIALIZATION ===")
        extractor = FinancialDataExtractorOpenAI()  # Using OpenAI extractor
        logger.info("Extractor initialized successfully")

        # Loop over each filing file in the list.
        logger.info("=== STEP 3: PROCESSING FILINGS ===")
        for file_index, file_path in enumerate(filing_files, 1):
            logger.info(f"\n--- Processing filing {file_index}/{len(filing_files)} ---")
            
            # Derive a base file name (expected format: <TICKER>-10k-<year>.txt).
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            logger.info(f"Processing filing: {file_path}")
            logger.info(f"Base file name: {file_name}")

            # Step 3.1: Clean the filing file
            logger.info("Step 3.1: Cleaning filing text...")
            clean_start = time.time()
            try:
                clean_text = DataCleaner(file_path)
                clean_text = str(clean_text)
                clean_time = time.time() - clean_start
                logger.info(f"Text cleaning completed in {clean_time:.2f} seconds")
                logger.info(f"Cleaned text length: {len(clean_text)} characters")
                
                # Define a path to save the cleaned text.
                processed_dir = os.path.join("data", "processed")
                os.makedirs(processed_dir, exist_ok=True)
                cleaned_file_path = os.path.join(processed_dir, f"{file_name}.txt")
                with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                logger.info(f"Cleaned text saved to: {cleaned_file_path}")
                
            except Exception as e:
                logger.error(f"Error cleaning text for {file_path}: {e}")
                continue

            # Set the query.
            query = "Consolidated Balance Sheets"
            logger.info(f"Extraction query: {query}")

            # Step 3.2: Extract financial data from the cleaned file
            logger.info("Step 3.2: Extracting financial data...")
            extraction_start = time.time()
            try:
                json_response = extractor.extract_from_file(cleaned_file_path, query)
                extraction_time = time.time() - extraction_start
                
                if json_response:
                    logger.info(f"JSON response generated successfully in {extraction_time:.2f} seconds")
                    logger.info(f"Response length: {len(json_response)} characters")
                else:
                    logger.warning("No JSON response generated; skipping file.")
                    continue
                    
            except Exception as e:
                logger.error(f"Error extracting data from {cleaned_file_path}: {e}")
                continue

            # Step 3.3: Process the JSON response into a flattened DataFrame
            logger.info("Step 3.3: Processing and flattening JSON response...")
            processing_start = time.time()
            try:
                df = extractor.process_and_flatten(json_response)
                processing_time = time.time() - processing_start
                
                # Reset index to make "Statement", "Year", and "Item" available as columns.
                df = df.reset_index()
                logger.info(f"DataFrame processing completed in {processing_time:.2f} seconds")
                logger.info(f"Flattened DataFrame shape: {df.shape}")
                logger.info("DataFrame columns:", list(df.columns))
                logger.info("\nDataFrame head:")
                logger.info(df.head().to_string())

                # Extract ticker and file year from the file name.
                # Expected file name format: <TICKER>-10k-<year>.txt
                parts = file_name.split('-')
                if len(parts) >= 3:
                    ticker_extracted = parts[0]
                    file_year = parts[2]
                else:
                    ticker_extracted = ""
                    file_year = ""

                # Add the ticker and file_year as new columns.
                df['ticker'] = ticker_extracted
                df['file_year'] = file_year
                logger.info(f"Added metadata - ticker: {ticker_extracted}, file_year: {file_year}")

                # Append this DataFrame to our list.
                final_dataframes.append(df)
                logger.info(f"DataFrame added to final collection (total: {len(final_dataframes)})")
                
            except Exception as e:
                logger.error(f"Error processing JSON response for {file_path}: {e}")
                continue
            
            file_total_time = clean_time + extraction_time + processing_time
            logger.info(f"--- Filing {file_index} processing complete in {file_total_time:.2f} seconds ---")

        # Step 4: Combine all processed DataFrames
        logger.info("\n=== STEP 4: COMBINING AND SAVING RESULTS ===")
        if final_dataframes:
            logger.info(f"Combining {len(final_dataframes)} DataFrames...")
            final_df = pd.concat(final_dataframes, ignore_index=True)
            logger.info(f"Combined DataFrame shape: {final_df.shape}")
            logger.info(f"Combined DataFrame columns: {list(final_df.columns)}")
            
            final_dir = os.path.join("data", "final")
            os.makedirs(final_dir, exist_ok=True)
            final_csv_path = os.path.join(final_dir, f"{ticker}_extracted.csv")
            final_df.to_csv(final_csv_path, index=False)
            logger.info(f"Combined DataFrame saved to: {final_csv_path}")
            
            # Log summary statistics
            logger.info("\n=== EXTRACTION SUMMARY ===")
            logger.info(f"Total filings processed: {len(final_dataframes)}")
            logger.info(f"Total rows extracted: {len(final_df)}")
            logger.info(f"Unique statements: {final_df['Statement'].nunique() if 'Statement' in final_df.columns else 'N/A'}")
            logger.info(f"Unique years: {final_df['Year'].nunique() if 'Year' in final_df.columns else 'N/A'}")
            logger.info(f"Unique items: {final_df['Item'].nunique() if 'Item' in final_df.columns else 'N/A'}")
            
        else:
            logger.warning("No data frames were processed successfully.")

    except Exception as e:
        logger.error(f"Critical error in main pipeline: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"=== PIPELINE COMPLETE - Total time: {total_time:.2f} seconds ===")
        logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
