import os
from openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import json
import pandas as pd
import sys
import concurrent.futures
import logging
import time
import math
from datetime import datetime

# Configure logging for detailed step tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/basic_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_with_timeout(func, timeout, *args, **kwargs):
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # If the call times out after 3 minutes, log and retry.
            logger.warning("Request timed out after 3 minutes. Retrying...")

def validate_environment():
    """Validate that OpenAI API key is present"""
    logger.info("=== STEP 1: VALIDATING ENVIRONMENT ===")
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        logger.error(f"Missing required environment variables: {', '.join(missing_keys)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    logger.info("Environment Variables Status:")
    for key in required_keys:
        logger.info(f"{key}: Found")
    logger.info("=== ENVIRONMENT VALIDATION COMPLETE ===\n")

# Validate environment variables at module import
validate_environment()

class FinancialDataExtractorOpenAI:
    def __init__(self):
        logger.info("=== INITIALIZING OPENAI EXTRACTOR ===")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI client configured successfully")
        logger.info("=== OPENAI EXTRACTOR INITIALIZATION COMPLETE ===\n")

    def extract_from_file(self, file_path, query):
        """
        Reads file content, constructs a detailed prompt with the provided query,
        generates an initial answer from the OpenAI ChatCompletion API,
        then uses that answer in a second prompt asking for a revised output,
        and returns the final revised output.
        Also logs JSON outputs with probabilities to CSV files.
        """
        logger.info(f"=== STEP 2: FILE EXTRACTION STARTED (OpenAI) ===")
        logger.info(f"File: {file_path}")
        logger.info(f"Query: {query}")
        
        # Create output directory for JSON logs
        json_logs_dir = os.path.join("data", "json_logs")
        os.makedirs(json_logs_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Step 2.1: Read file content
        logger.info("Step 2.1: Reading file content...")
        start_time = time.time()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                passage = f.read()
            read_time = time.time() - start_time
            logger.info(f"File read successfully in {read_time:.2f} seconds")
            logger.info(f"Content length: {len(passage)} characters")
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise

        # Convert multiline text to a one-line string for the prompt.
        passage_oneline = passage.replace("\n", " ")
        query_oneline = query.replace("\n", " ")
        logger.info(f"Text preprocessing complete. Final length: {len(passage_oneline)} characters")

        # Step 2.2: First OpenAI call with logprobs
        logger.info("Step 2.2: Making initial OpenAI call with probability tracking...")
        initial_prompt = f""" 
            
            You are trying to extract financial information from a document. This PASSAGE contains SEC 10-K statements for a company presented as markdown tables.

            Extract the financial information related to the QUESTION and provide the output in a specific JSON format following these exact requirements:

            The output must be valid, parseable JSON with no markdown code blocks, no extra text, and no additional formatting.

            The top-level keys should be financial statement types (e.g., "Balance Sheet", "Income Statement").

            Each statement type should contain year objects (e.g., "2023", "2022").

            Each year should contain line items (e.g., "Revenue", "Net Income").

            Each line item should be an object with the following required keys:

            "value": The numeric value (keep as a string if it includes special formatting), if the values in inside () make it negative.

            "unit": The unit of measurement (e.g., "USD millions", "Billion", etc.)

            Example structure: {{ "Balance Sheet": {{ "2023": {{ "Total Assets": {{ "value": "100,000", "unit": "million USD" }}, "Total Liabilities": {{ "value": "45,000", "unit": "thousand USD" }} }}, "2022": {{ "Total Assets": {{ "value": "90", "unit": "billion USD" }} }} }} }}

            Extract ALL relevant financial figures for ALL available years from the passage. Ignore any markdown table formatting and extract only the raw financial data. Do not include any explanatory text, markdown formatting, or code block indicators in your output.

            QUESTION: {query_oneline} 
            PASSAGE: {passage_oneline}
            
            """

        logger.info(f"Prompt length: {len(initial_prompt)} characters")
        
        try:
            start_time = time.time()
            response_initial = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0,
                logprobs=True,
                top_logprobs=5
            )
            first_call_time = time.time() - start_time
            first_output = response_initial.choices[0].message.content
            
            # Extract and log probability information
            first_call_data = {
                'timestamp': timestamp,
                'file_name': file_base_name,
                'call_type': 'initial_call',
                'response_text': first_output,
                'response_length': len(first_output),
                'call_time_seconds': first_call_time,
                'model': "gpt-4o-mini",
                'temperature': 0
            }
            
            if response_initial.choices[0].logprobs and response_initial.choices[0].logprobs.content:
                logprobs = [token.logprob for token in response_initial.choices[0].logprobs.content]
                avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
                avg_probability = math.exp(avg_logprob) if avg_logprob is not None else None
                
                first_call_data.update({
                    'avg_log_probability': avg_logprob,
                    'avg_probability': avg_probability,
                    'token_count': len(logprobs),
                    'min_log_probability': min(logprobs) if logprobs else None,
                    'max_log_probability': max(logprobs) if logprobs else None
                })
                
                logger.info(f"Initial LLM call completed in {first_call_time:.2f} seconds")
                logger.info(f"Initial response length: {len(first_output)} characters")
                logger.info(f"Average log probability: {avg_logprob:.4f}" if avg_logprob else "Average log probability: N/A")
                logger.info(f"Average probability: {avg_probability:.4f}" if avg_probability else "Average probability: N/A")
                logger.info(f"Number of tokens with probabilities: {len(logprobs)}")
                
                # Log confidence assessment
                if avg_probability:
                    if avg_probability > 0.8:
                        confidence_level = "HIGH"
                    elif avg_probability > 0.5:
                        confidence_level = "MEDIUM"
                    else:
                        confidence_level = "LOW"
                    
                    first_call_data['confidence_level'] = confidence_level
                    logger.info(f"Confidence level: {confidence_level}")
                    
                    if avg_probability < 0.5:
                        logger.warning(f"⚠️ Low confidence extraction: {avg_probability:.4f}")
            else:
                first_call_data.update({
                    'avg_log_probability': None,
                    'avg_probability': None,
                    'token_count': 0,
                    'confidence_level': 'UNKNOWN'
                })
                        
            logger.info(f"Initial response preview: {first_output[:200]}...")
            
        except Exception as e:
            logger.error(f"Error in initial OpenAI call: {e}")
            raise

        # Construct a second prompt using the first output.
        logger.info("Step 2.3: Making revision OpenAI call...")
        second_prompt = f"""
            The following JSON output was generated based on the provided financial statements:
            {first_output}
            
            Please revise the output to ensure it strictly adheres to the following JSON format requirements and includes any missing relevant financial data:
            
            1. Valid, parseable JSON with no markdown code blocks, no extra text, and no additional formatting.
            2. Top-level keys are financial statement types (e.g., "Balance Sheet", "Income Statement").
            3. Each statement type contains year objects (e.g., "2023", "2022").
            4. Each year contains line items (e.g., "Revenue", "Net Income").
            5. Each line item is an object with the keys "value" and "unit".
            
            QUESTION: {query_oneline} 
            PASSAGE: {passage_oneline}

            Revised JSON:
            """

        try:
            start_time = time.time()
            response_revised = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": second_prompt}],
                temperature=0,
                logprobs=True,
                top_logprobs=5
            )
            second_call_time = time.time() - start_time
            final_output = response_revised.choices[0].message.content
            
            # Extract and log revision probability information
            second_call_data = {
                'timestamp': timestamp,
                'file_name': file_base_name,
                'call_type': 'revision_call',
                'response_text': final_output,
                'response_length': len(final_output),
                'call_time_seconds': second_call_time,
                'model': "gpt-4o-mini",
                'temperature': 0
            }
            
            if response_revised.choices[0].logprobs and response_revised.choices[0].logprobs.content:
                logprobs = [token.logprob for token in response_revised.choices[0].logprobs.content]
                avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
                avg_probability = math.exp(avg_logprob) if avg_logprob is not None else None
                
                second_call_data.update({
                    'avg_log_probability': avg_logprob,
                    'avg_probability': avg_probability,
                    'token_count': len(logprobs),
                    'min_log_probability': min(logprobs) if logprobs else None,
                    'max_log_probability': max(logprobs) if logprobs else None
                })
                
                logger.info(f"Revision LLM call completed in {second_call_time:.2f} seconds")
                logger.info(f"Final response length: {len(final_output)} characters")
                logger.info(f"Revision average log probability: {avg_logprob:.4f}" if avg_logprob else "Revision average log probability: N/A")
                logger.info(f"Revision average probability: {avg_probability:.4f}" if avg_probability else "Revision average probability: N/A")
                logger.info(f"Revision tokens with probabilities: {len(logprobs)}")
                
                # Log revision confidence assessment
                if avg_probability:
                    if avg_probability > 0.8:
                        confidence_level = "HIGH"
                    elif avg_probability > 0.5:
                        confidence_level = "MEDIUM"
                    else:
                        confidence_level = "LOW"
                    
                    second_call_data['confidence_level'] = confidence_level
                    logger.info(f"Revision confidence level: {confidence_level}")
                    
                    if avg_probability < 0.5:
                        logger.warning(f"⚠️ Low confidence revision: {avg_probability:.4f}")
            else:
                second_call_data.update({
                    'avg_log_probability': None,
                    'avg_probability': None,
                    'token_count': 0,
                    'confidence_level': 'UNKNOWN'
                })
                        
            logger.info(f"Final response preview: {final_output[:200]}...")
            
        except Exception as e:
            logger.error(f"Error in revision OpenAI call: {e}")
            raise

        # Save JSON outputs with probabilities to CSV
        logger.info("Step 2.4: Saving JSON outputs with probabilities to CSV...")
        try:
            # Combine both calls data
            json_responses_data = [first_call_data, second_call_data]
            json_df = pd.DataFrame(json_responses_data)
            
            # Save to CSV
            csv_filename = f"json_responses_{file_base_name}_{timestamp}.csv"
            csv_path = os.path.join(json_logs_dir, csv_filename)
            json_df.to_csv(csv_path, index=False)
            
            logger.info(f"JSON responses with probabilities saved to: {csv_path}")
            logger.info(f"CSV contains {len(json_df)} responses (initial + revision)")
            
            # Log summary of probabilities
            if 'avg_probability' in json_df.columns:
                avg_probs = json_df['avg_probability'].dropna()
                if not avg_probs.empty:
                    logger.info(f"Probability summary - Mean: {avg_probs.mean():.4f}, Min: {avg_probs.min():.4f}, Max: {avg_probs.max():.4f}")
            
        except Exception as e:
            logger.error(f"Error saving JSON responses to CSV: {e}")
            # Don't raise here as this is just logging functionality

        total_time = first_call_time + second_call_time + read_time
        logger.info(f"=== EXTRACTION COMPLETE (OpenAI) - Total time: {total_time:.2f} seconds ===\n")
        
        return final_output

    def process_and_flatten(self, json_text):
        """
        Cleans the JSON text, converts it into a dictionary, and then
        flattens the nested JSON structure into a pandas DataFrame.
        """
        logger.info("=== STEP 3: JSON PROCESSING AND FLATTENING (OpenAI) ===")
        logger.info(f"Input JSON length: {len(json_text)} characters")
        
        # Step 3.1: Clean the JSON text
        logger.info("Step 3.1: Cleaning JSON text...")
        try:
            lines = json_text.splitlines()
            json_lines = [line for line in lines if not line.startswith("```")]
            cleaned_text = "\n".join(json_lines).strip()
            logger.info(f"JSON cleaning complete. Cleaned length: {len(cleaned_text)} characters")
            logger.info(f"Removed {len(lines) - len(json_lines)} markdown lines")
        except Exception as e:
            logger.error(f"Error cleaning JSON text: {e}")
            raise

        # Step 3.2: Parse JSON
        logger.info("Step 3.2: Parsing JSON...")
        try:
            data = json.loads(cleaned_text)
            logger.info("JSON parsing successful")
            logger.info(f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Problematic JSON text: {cleaned_text[:500]}...")
            raise

        # Step 3.3: Flatten the structure
        logger.info("Step 3.3: Flattening nested JSON structure...")
        try:
            flattened_data = []
            statements_processed = 0
            items_processed = 0
            
            for statement, years in data.items():
                statements_processed += 1
                logger.info(f"Processing statement: {statement}")
                
                if not isinstance(years, dict):
                    logger.warning(f"Unexpected data type for statement {statement}: {type(years)}")
                    continue
                    
                for year, items in years.items():
                    logger.info(f"  Processing year: {year}")
                    
                    if not isinstance(items, dict):
                        logger.warning(f"Unexpected data type for year {year}: {type(items)}")
                        continue
                        
                    for item, values in items.items():
                        items_processed += 1
                        row = {
                            'Statement': statement,
                            'Year': year,
                            'Item': item,
                            **values  # Unpacks the "value" and "unit" keys into the row
                        }
                        flattened_data.append(row)
                        
            logger.info(f"Flattening complete: {statements_processed} statements, {items_processed} items processed")
            
        except Exception as e:
            logger.error(f"Error flattening JSON structure: {e}")
            raise

        # Step 3.4: Create DataFrame
        logger.info("Step 3.4: Creating pandas DataFrame...")
        try:
            df = pd.DataFrame(flattened_data)
            logger.info(f"DataFrame created with shape: {df.shape}")
            
            if not df.empty:
                logger.info(f"DataFrame columns: {list(df.columns)}")
                df = df.set_index(['Statement', 'Year', 'Item'])
                logger.info("DataFrame indexed successfully")
                
                # Log summary statistics
                logger.info(f"Unique statements: {df.index.get_level_values('Statement').nunique()}")
                logger.info(f"Unique years: {df.index.get_level_values('Year').nunique()}")
                logger.info(f"Unique items: {df.index.get_level_values('Item').nunique()}")
            else:
                logger.warning("Empty DataFrame created")
                
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            raise
            
        logger.info("=== JSON PROCESSING AND FLATTENING COMPLETE (OpenAI) ===\n")
        return df
