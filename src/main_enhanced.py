#!/usr/bin/env python3
"""
Enhanced Main Pipeline with Comprehensive Logging and Probability Tracking

This script uses the EnhancedFinancialExtractor to provide detailed step-by-step
logging and probability tracking for each extraction operation.
"""

from downloader import Config, SECDataLoader
from preprocessor import DataCleaner
from enhanced_extractor import EnhancedFinancialExtractor
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
        logging.FileHandler('logs/enhanced_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main enhanced pipeline function."""
    logger.info("="*80)
    logger.info("=== ENHANCED FINANCIAL DATA EXTRACTION PIPELINE STARTED ===")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # Initialize configuration and SECDataLoader
        logger.info("=== STEP 1: INITIALIZATION ===")
        config = Config()
        sec_loader = SECDataLoader(config)
        logger.info("SEC Data Loader initialized successfully")
        
        # Configuration
        form_type = "10-K"
        ticker = "NVDA"
        cutoff = 1  # Only process one year of reports
        model = "gpt-4o-mini"
        confidence_threshold = 0.7
        
        logger.info(f"Configuration:")
        logger.info(f"  Ticker: {ticker}")
        logger.info(f"  Form type: {form_type}")
        logger.info(f"  Cutoff: {cutoff} (single year only)")
        logger.info(f"  Model: {model}")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        
        # Step 1.1: Download SEC filings
        logger.info("Step 1.1: Downloading SEC filings...")
        download_start = time.time()
        download_dir = sec_loader.download_sec_filing(form_type, ticker)
        download_time = time.time() - download_start
        logger.info(f"Filings downloaded to: {download_dir} (took {download_time:.2f} seconds)")

        # Step 1.2: Create downloaded forms DataFrame
        logger.info("Step 1.2: Creating downloaded forms DataFrame...")
        downloaded_forms_df = sec_loader.get_downloaded_forms_df()
        final_dir = os.path.join("data", "enhanced_final")
        os.makedirs(final_dir, exist_ok=True)
        forms_csv_path = os.path.join(final_dir, "downloaded_forms.csv")
        downloaded_forms_df.to_csv(forms_csv_path, index=False)
        logger.info(f"Downloaded forms information saved to: {forms_csv_path}")
        logger.info(f"Downloaded forms shape: {downloaded_forms_df.shape}")
        
        # Step 1.3: Rename and copy filings
        logger.info("Step 1.3: Renaming and copying filings...")
        sec_loader.rename_and_copy_filings()
        logger.info("Filing rename and copy complete")
        
        # Step 1.4: List downloaded filings
        logger.info("Step 1.4: Listing downloaded filings...")
        filing_files = sec_loader.list_downloaded_filings(ticker, form_type, cutoff)
        if not filing_files:
            logger.error("No filings found in the downloaded directory.")
            return
        
        logger.info(f"Found {len(filing_files)} filing files:")
        for i, file_path in enumerate(filing_files, 1):
            logger.info(f"  {i}. {file_path}")

        # Step 2: Initialize Enhanced Extractor
        logger.info("=== STEP 2: ENHANCED EXTRACTOR INITIALIZATION ===")
        extractor = EnhancedFinancialExtractor(model=model)
        logger.info("Enhanced extractor initialized successfully")

        # Step 3: Process filings with enhanced extraction
        logger.info("=== STEP 3: ENHANCED FILING PROCESSING ===")
        
        # Initialize collections for results and statistics
        final_dataframes = []
        extraction_stats = []
        
        for file_index, file_path in enumerate(filing_files, 1):
            logger.info(f"\n{'-'*60}")
            logger.info(f"Processing filing {file_index}/{len(filing_files)}")
            logger.info(f"{'-'*60}")
            
            # Derive file metadata
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
                logger.info(f"Cleaned text length: {len(clean_text):,} characters")
                
                # Save cleaned text
                processed_dir = os.path.join("data", "enhanced_processed")
                os.makedirs(processed_dir, exist_ok=True)
                cleaned_file_path = os.path.join(processed_dir, f"{file_name}.txt")
                with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                logger.info(f"Cleaned text saved to: {cleaned_file_path}")
                
            except Exception as e:
                logger.error(f"Error cleaning text for {file_path}: {e}")
                extraction_stats.append({
                    'file_name': file_name,
                    'status': 'failed_cleaning',
                    'error': str(e)
                })
                continue

            # Step 3.2: Enhanced extraction with separate call outputs
            logger.info("Step 3.2: Enhanced financial data extraction...")
            query = "Consolidated Balance Sheets"
            logger.info(f"Extraction query: {query}")
            
            extraction_start = time.time()
            try:
                df1, df2 = extractor.extract_from_file_both_calls(
                    cleaned_file_path, 
                    query, 
                    confidence_threshold=confidence_threshold
                )
                extraction_time = time.time() - extraction_start
                
                if df1.empty and df2.empty:
                    logger.warning("Both DataFrames returned empty from extraction")
                    extraction_stats.append({
                        'file_name': file_name,
                        'status': 'empty_result',
                        'extraction_time': extraction_time
                    })
                    continue
                
                logger.info(f"Extraction completed in {extraction_time:.2f} seconds")
                logger.info(f"Call 1 DataFrame shape: {df1.shape}")
                logger.info(f"Call 2 DataFrame shape: {df2.shape}")
                
                # Add file metadata to both DataFrames
                parts = file_name.split('-')
                if len(parts) >= 3:
                    ticker_extracted = parts[0]
                    file_year = parts[2]
                else:
                    ticker_extracted = ticker
                    file_year = "unknown"

                # Process Call 1 DataFrame
                if not df1.empty:
                    df1['ticker'] = ticker_extracted
                    df1['file_year'] = file_year
                    df1['file_index'] = file_index
                    df1['extraction_time'] = extraction_time
                    logger.info(f"Added metadata to Call 1 - ticker: {ticker_extracted}, file_year: {file_year}")

                # Process Call 2 DataFrame
                if not df2.empty:
                    df2['ticker'] = ticker_extracted
                    df2['file_year'] = file_year
                    df2['file_index'] = file_index
                    df2['extraction_time'] = extraction_time
                    logger.info(f"Added metadata to Call 2 - ticker: {ticker_extracted}, file_year: {file_year}")

                # Collect statistics for both calls
                stats = {
                    'file_name': file_name,
                    'status': 'success',
                    'call1_rows': len(df1),
                    'call2_rows': len(df2),
                    'extraction_time': extraction_time,
                    'ticker': ticker_extracted,
                    'file_year': file_year
                }
                
                # Add probability statistics for both calls
                if not df1.empty and 'model_probability' in df1.columns:
                    stats.update({
                        'call1_avg_probability': df1['model_probability'].mean(),
                        'call1_min_probability': df1['model_probability'].min(),
                        'call1_max_probability': df1['model_probability'].max(),
                        'call1_low_confidence_count': (df1['model_probability'] < confidence_threshold).sum(),
                        'call1_high_confidence_count': (df1['model_probability'] > 0.8).sum()
                    })
                
                if not df2.empty and 'model_probability' in df2.columns:
                    stats.update({
                        'call2_avg_probability': df2['model_probability'].mean(),
                        'call2_min_probability': df2['model_probability'].min(),
                        'call2_max_probability': df2['model_probability'].max(),
                        'call2_low_confidence_count': (df2['model_probability'] < confidence_threshold).sum(),
                        'call2_high_confidence_count': (df2['model_probability'] > 0.8).sum()
                    })
                
                # Add counts for unique elements in both calls
                if not df1.empty:
                    if 'Statement' in df1.columns:
                        stats['call1_unique_statements'] = df1['Statement'].nunique()
                    if 'Year' in df1.columns:
                        stats['call1_unique_years'] = df1['Year'].nunique()
                    if 'Item' in df1.columns:
                        stats['call1_unique_items'] = df1['Item'].nunique()
                
                if not df2.empty:
                    if 'Statement' in df2.columns:
                        stats['call2_unique_statements'] = df2['Statement'].nunique()
                    if 'Year' in df2.columns:
                        stats['call2_unique_years'] = df2['Year'].nunique()
                    if 'Item' in df2.columns:
                        stats['call2_unique_items'] = df2['Item'].nunique()
                
                extraction_stats.append(stats)
                
                # Store DataFrames separately for call1 and call2
                if not df1.empty:
                    final_dataframes.append(('call1', df1))
                    logger.info(f"Call 1 DataFrame added to final collection")
                
                if not df2.empty:
                    final_dataframes.append(('call2', df2))
                    logger.info(f"Call 2 DataFrame added to final collection")
                
                logger.info(f"Total DataFrames in collection: {len(final_dataframes)}")
                
            except Exception as e:
                logger.error(f"Error in enhanced extraction for {file_path}: {e}")
                extraction_stats.append({
                    'file_name': file_name,
                    'status': 'failed_extraction',
                    'error': str(e)
                })
                continue
            
            file_total_time = clean_time + extraction_time
            logger.info(f"Filing {file_index} processing complete in {file_total_time:.2f} seconds")

        # Step 4: Combine and analyze results
        logger.info("\n" + "="*80)
        logger.info("=== STEP 4: RESULTS COMBINATION AND ANALYSIS ===")
        logger.info("="*80)
        
        if final_dataframes:
            logger.info(f"Combining {len(final_dataframes)} DataFrames...")
            
            # Separate call1 and call2 DataFrames
            call1_dataframes = [df for call_type, df in final_dataframes if call_type == 'call1']
            call2_dataframes = [df for call_type, df in final_dataframes if call_type == 'call2']
            
            logger.info(f"Call 1 DataFrames: {len(call1_dataframes)}")
            logger.info(f"Call 2 DataFrames: {len(call2_dataframes)}")
            
            # Create separate directories for call1 and call2
            call1_dir = os.path.join(final_dir, "call1")
            call2_dir = os.path.join(final_dir, "call2")
            os.makedirs(call1_dir, exist_ok=True)
            os.makedirs(call2_dir, exist_ok=True)
            
            # Combine and save call1 results
            if call1_dataframes:
                call1_df = pd.concat(call1_dataframes, ignore_index=True)
                logger.info(f"Call 1 combined DataFrame shape: {call1_df.shape}")
                call1_csv_path = os.path.join(call1_dir, f"{ticker}_call1_extraction.csv")
                call1_df.to_csv(call1_csv_path, index=False)
                logger.info(f"Call 1 DataFrame saved to: {call1_csv_path}")
            
            # Combine and save call2 results
            if call2_dataframes:
                call2_df = pd.concat(call2_dataframes, ignore_index=True)
                logger.info(f"Call 2 combined DataFrame shape: {call2_df.shape}")
                call2_csv_path = os.path.join(call2_dir, f"{ticker}_call2_extraction.csv")
                call2_df.to_csv(call2_csv_path, index=False)
                logger.info(f"Call 2 DataFrame saved to: {call2_csv_path}")
            
            # Also create combined file for backward compatibility
            final_df = pd.concat([df for _, df in final_dataframes], ignore_index=True)
            logger.info(f"Combined DataFrame shape: {final_df.shape}")
            logger.info(f"Combined DataFrame columns: {list(final_df.columns)}")
            
            # Save combined results
            final_csv_path = os.path.join(final_dir, f"{ticker}_enhanced_extraction.csv")
            final_df.to_csv(final_csv_path, index=False)
            logger.info(f"Combined DataFrame saved to: {final_csv_path}")
            
            # Generate and save extraction statistics
            stats_df = pd.DataFrame(extraction_stats)
            stats_csv_path = os.path.join(final_dir, f"{ticker}_extraction_statistics.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            logger.info(f"Extraction statistics saved to: {stats_csv_path}")
            
            # Log comprehensive summary
            logger.info("\n" + "="*80)
            logger.info("=== COMPREHENSIVE EXTRACTION SUMMARY ===")
            logger.info("="*80)
            
            successful_extractions = len([s for s in extraction_stats if s['status'] == 'success'])
            failed_extractions = len(extraction_stats) - successful_extractions
            
            logger.info(f"Total filings processed: {len(extraction_stats)}")
            logger.info(f"Successful extractions: {successful_extractions}")
            logger.info(f"Failed extractions: {failed_extractions}")
            logger.info(f"Success rate: {successful_extractions/len(extraction_stats)*100:.1f}%")
            
            logger.info(f"\nData Quality Metrics:")
            logger.info(f"Total rows extracted (all calls): {len(final_df)}")
            if call1_dataframes:
                logger.info(f"Call 1 total rows: {len(call1_df)}")
            if call2_dataframes:
                logger.info(f"Call 2 total rows: {len(call2_df)}")
            
            logger.info(f"Unique statements: {final_df['Statement'].nunique() if 'Statement' in final_df.columns else 'N/A'}")
            logger.info(f"Unique years: {final_df['Year'].nunique() if 'Year' in final_df.columns else 'N/A'}")
            logger.info(f"Unique items: {final_df['Item'].nunique() if 'Item' in final_df.columns else 'N/A'}")
            
            # Probability analysis (if available)
            if 'model_probability' in final_df.columns:
                avg_prob = final_df['model_probability'].mean()
                logger.info(f"\nProbability Analysis (Combined):")
                logger.info(f"Average model probability: {avg_prob:.4f}")
                logger.info(f"Min probability: {final_df['model_probability'].min():.4f}")
                logger.info(f"Max probability: {final_df['model_probability'].max():.4f}")
                
                high_conf = (final_df['model_probability'] > 0.8).sum()
                med_conf = ((final_df['model_probability'] > 0.5) & (final_df['model_probability'] <= 0.8)).sum()
                low_conf = (final_df['model_probability'] <= 0.5).sum()
                
                logger.info(f"High confidence rows (>0.8): {high_conf} ({high_conf/len(final_df)*100:.1f}%)")
                logger.info(f"Medium confidence rows (0.5-0.8): {med_conf} ({med_conf/len(final_df)*100:.1f}%)")
                logger.info(f"Low confidence rows (≤0.5): {low_conf} ({low_conf/len(final_df)*100:.1f}%)")
                
                if low_conf > 0:
                    logger.warning(f"⚠️ {low_conf} rows require manual review due to low confidence")
                
                # Separate analysis for each call
                if call1_dataframes and 'model_probability' in call1_df.columns:
                    logger.info(f"\nCall 1 Probability Analysis:")
                    logger.info(f"Average probability: {call1_df['model_probability'].mean():.4f}")
                    logger.info(f"High confidence rows: {(call1_df['model_probability'] > 0.8).sum()}")
                
                if call2_dataframes and 'model_probability' in call2_df.columns:
                    logger.info(f"\nCall 2 Probability Analysis:")
                    logger.info(f"Average probability: {call2_df['model_probability'].mean():.4f}")
                    logger.info(f"High confidence rows: {(call2_df['model_probability'] > 0.8).sum()}")
            
            # Performance analysis
            successful_stats = [s for s in extraction_stats if s['status'] == 'success']
            if successful_stats:
                avg_extraction_time = sum(s['extraction_time'] for s in successful_stats) / len(successful_stats)
                total_extraction_time = sum(s['extraction_time'] for s in successful_stats)
                
                logger.info(f"\nPerformance Metrics:")
                logger.info(f"Average extraction time per file: {avg_extraction_time:.2f} seconds")
                logger.info(f"Total extraction time: {total_extraction_time:.2f} seconds")
                logger.info(f"Average rows per second: {len(final_df)/total_extraction_time:.2f}")
            
        else:
            logger.warning("No data frames were processed successfully.")
            logger.warning("Check the extraction statistics for details on failures.")

    except Exception as e:
        logger.error(f"Critical error in enhanced pipeline: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"=== ENHANCED PIPELINE COMPLETE - Total time: {total_time:.2f} seconds ===")
        logger.info(f"{'='*80}")
        
        # Log file locations
        logger.info(f"\n📋 Log files generated:")
        logger.info(f"  - logs/enhanced_pipeline.log (main pipeline)")
        logger.info(f"  - logs/enhanced_extraction.log (detailed extraction)")
        
        logger.info(f"\n📁 Output directory: {final_dir}")
        logger.info(f"  - call1/ folder: Initial extraction results")
        logger.info(f"  - call2/ folder: Refined extraction results")
        logger.info(f"  - Combined results in main directory")

if __name__ == "__main__":
    main() 