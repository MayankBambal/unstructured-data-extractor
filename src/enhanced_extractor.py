import os
import re
import json
import math
import time
import logging
import concurrent.futures
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, AuthenticationError, OpenAIError
from datetime import datetime

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedFinancialExtractor:
    """
    Enhanced financial data extractor with comprehensive logging, 
    probability tracking, and robust error handling.
    Supports OpenAI only.
    """
    
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the enhanced extractor.
        
        Args:
            model (str): OpenAI model to use
        """
        logger.info("=== INITIALIZING ENHANCED FINANCIAL EXTRACTOR ===")
        self.provider = "openai"
        self.model = model
        
        self._init_openai()
        
        logger.info(f"Enhanced extractor initialized with OpenAI ({self.model})")
        logger.info("=== INITIALIZATION COMPLETE ===\n")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise EnvironmentError('Missing OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI client configured successfully")
    
    def run_with_timeout(self, func, timeout: int = 180, max_retries: int = 3):
        """Execute function with timeout and exponential backoff retries."""
        delay = 5
        for attempt in range(max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor() as exe:
                    future = exe.submit(func)
                    result = future.result(timeout=timeout)
                    if attempt > 0:
                        logger.info(f"Successfully completed after {attempt + 1} attempts")
                    return result
            except concurrent.futures.TimeoutError:
                if attempt == max_retries - 1:
                    logger.error(f"Function timed out after {max_retries} attempts")
                    raise
                logger.warning(f'Timeout on attempt {attempt+1}/{max_retries}; retrying in {delay}s...')
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Function failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f'Error on attempt {attempt+1}/{max_retries}: {e}; retrying in {delay}s...')
                time.sleep(delay)
                delay *= 2
    
    def clean_and_repair_json(self, raw: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Advanced JSON cleaning and repair with detailed logging.
        """
        logger.info("Step: Cleaning and repairing JSON response...")
        logger.info(f"Raw response length: {len(raw)} characters")
        
        # Step 1: Remove code fences
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        logger.info(f"After removing code fences: {len(text)} characters")
        
        # Step 2: Try direct parse
        try:
            result = json.loads(text)
            logger.info("✅ JSON parsed successfully on first attempt")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}")
        
        # Step 3: Extract JSON object/array
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end < 0:
            # Try array format
            start, end = text.find('['), text.rfind(']')
            if start < 0 or end < 0:
                raise ValueError('No JSON object or array found')
        
        blob = text[start:end+1]
        logger.info(f"Extracted JSON blob: {len(blob)} characters")
        
        # Step 4: Remove trailing commas
        original_blob = blob
        blob = re.sub(r',\s*([}\]])', r'\1', blob)
        if blob != original_blob:
            logger.info("Removed trailing commas")
        
        # Step 5: Wrap multiple objects in array if needed
        if re.search(r'}\s*,\s*{', blob) and not blob.strip().startswith('['):
            blob = f'[{blob}]'
            logger.info("Wrapped multiple objects in array")
        
        # Step 6: Try parse cleaned blob
        try:
            result = json.loads(blob)
            logger.info("✅ JSON parsed successfully after cleaning")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f'JSON parse failed after cleanup: {e}')
            
            # Step 7: Balance braces and brackets
            o_c, c_c = blob.count('{'), blob.count('}')
            o_s, c_s = blob.count('['), blob.count(']')
            
            if o_c != c_c:
                blob += '}' * max(0, o_c - c_c)
                logger.info(f"Added {max(0, o_c - c_c)} closing braces")
            
            if o_s != c_s:
                blob += ']' * max(0, o_s - c_s)
                logger.info(f"Added {max(0, o_s - c_s)} closing brackets")
            
            result = json.loads(blob)
            logger.info("✅ JSON parsed successfully after brace balancing")
            return result
    
    def extract_with_openai(self, passage: str, query: str, file_base_name: str = "unknown") -> Tuple[str, str, Dict[str, float]]:
        """Extract using OpenAI with probability tracking and CSV logging. Returns both call responses."""
        logger.info("Making OpenAI API calls...")
        
        # Create output directory for JSON logs
        json_logs_dir = os.path.join("data", "enhanced_json_logs")
        os.makedirs(json_logs_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        prompt = f"""
        You are extracting financial figures from SEC 10-K markdown tables.
        Provide ALL relevant data as a JSON array of objects with EXACT keys:
        - "Statement" (e.g., "Balance Sheet")
        - "Item" (line item name)  
        - "Year" (four-digit year)
        - "value" (numeric string; parentheses for negatives)
        - "unit" (unit of measure, e.g., "USD millions")
        
        The output must be valid JSON (no code fences or extra text).
        
        QUERY: {query}
        PASSAGE: {passage}
        """
        
        def api_call_1():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                logprobs=True,
                top_logprobs=5
            )
        
        # First API call
        start_time = time.time()
        resp1 = self.run_with_timeout(api_call_1, timeout=180)
        call1_time = time.time() - start_time
        
        raw1 = resp1.choices[0].message.content
        logger.info(f"First API call completed in {call1_time:.2f}s")
        logger.info(f"Response length: {len(raw1)} characters")
        
        # Extract probabilities from first call and prepare CSV data
        first_call_data = {
            'timestamp': timestamp,
            'file_name': file_base_name,
            'call_type': 'initial_call',
            'response_text': raw1,
            'response_length': len(raw1),
            'call_time_seconds': call1_time,
            'model': self.model,
            'temperature': 0
        }
        
        probs1 = {}
        if resp1.choices[0].logprobs and resp1.choices[0].logprobs.content:
            logprobs = [tok.logprob for tok in resp1.choices[0].logprobs.content]
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                avg_prob = math.exp(avg_logprob)
                probs1 = {
                    'avg_logprob_1': avg_logprob,
                    'avg_probability_1': avg_prob,
                    'token_count_1': len(logprobs)
                }
                
                first_call_data.update({
                    'avg_log_probability': avg_logprob,
                    'avg_probability': avg_prob,
                    'token_count': len(logprobs),
                    'min_log_probability': min(logprobs),
                    'max_log_probability': max(logprobs)
                })
                
                confidence = "HIGH" if avg_prob > 0.8 else "MEDIUM" if avg_prob > 0.5 else "LOW"
                first_call_data['confidence_level'] = confidence
        else:
            first_call_data.update({
                'avg_log_probability': None,
                'avg_probability': None,
                'token_count': 0,
                'confidence_level': 'UNKNOWN'
            })
        
        # Second API call for refinement
        refinement_prompt = f"""
        Review and refine this JSON output for completeness and accuracy:
        {raw1}
        
        Ensure all financial line items from the passage are included with proper formatting.
        Return only the refined JSON array.
        
        ORIGINAL QUERY: {query}
        ORIGINAL PASSAGE: {passage}
        """
        
        def api_call_2():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': refinement_prompt}],
                temperature=0,
                logprobs=True,
                top_logprobs=5
            )
        
        start_time = time.time()
        resp2 = self.run_with_timeout(api_call_2, timeout=180)
        call2_time = time.time() - start_time
        
        raw2 = resp2.choices[0].message.content
        logger.info(f"Second API call completed in {call2_time:.2f}s")
        
        # Extract probabilities from second call and prepare CSV data
        second_call_data = {
            'timestamp': timestamp,
            'file_name': file_base_name,
            'call_type': 'refinement_call',
            'response_text': raw2,
            'response_length': len(raw2),
            'call_time_seconds': call2_time,
            'model': self.model,
            'temperature': 0
        }
        
        probs2 = {}
        if resp2.choices[0].logprobs and resp2.choices[0].logprobs.content:
            logprobs = [tok.logprob for tok in resp2.choices[0].logprobs.content]
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                avg_prob = math.exp(avg_logprob)
                probs2 = {
                    'avg_logprob_2': avg_logprob,
                    'avg_probability_2': avg_prob,
                    'token_count_2': len(logprobs)
                }
                
                second_call_data.update({
                    'avg_log_probability': avg_logprob,
                    'avg_probability': avg_prob,
                    'token_count': len(logprobs),
                    'min_log_probability': min(logprobs),
                    'max_log_probability': max(logprobs)
                })
                
                confidence = "HIGH" if avg_prob > 0.8 else "MEDIUM" if avg_prob > 0.5 else "LOW"
                second_call_data['confidence_level'] = confidence
        else:
            second_call_data.update({
                'avg_log_probability': None,
                'avg_probability': None,
                'token_count': 0,
                'confidence_level': 'UNKNOWN'
            })
        
        # Save JSON outputs with probabilities to CSV
        logger.info("Saving enhanced JSON outputs with probabilities to CSV...")
        try:
            # Combine both calls data
            json_responses_data = [first_call_data, second_call_data]
            json_df = pd.DataFrame(json_responses_data)
            
            # Save to CSV
            csv_filename = f"enhanced_json_responses_{file_base_name}_{timestamp}.csv"
            csv_path = os.path.join(json_logs_dir, csv_filename)
            json_df.to_csv(csv_path, index=False)
            
            logger.info(f"Enhanced JSON responses with probabilities saved to: {csv_path}")
            logger.info(f"CSV contains {len(json_df)} responses (initial + refinement)")
            
            # Log only final probability
            if probs2 and 'avg_probability_2' in probs2:
                logger.info(f"Final probability: {probs2['avg_probability_2']:.4f}")
            elif probs1 and 'avg_probability_1' in probs1:
                logger.info(f"Final probability: {probs1['avg_probability_1']:.4f}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced JSON responses to CSV: {e}")
            # Don't raise here as this is just logging functionality
        
        # Combine probability metrics
        all_probs = {**probs1, **probs2, 'total_time': call1_time + call2_time}
        
        return raw1, raw2, all_probs
    
    def extract_from_file_both_calls(self, file_path: str, query: str, confidence_threshold: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract financial data from file with enhanced error handling and probability tracking.
        Returns separate DataFrames for both API calls.
        
        Args:
            file_path: Path to the text file
            query: Extraction query
            confidence_threshold: Minimum confidence for automatic acceptance
            
        Returns:
            Tuple of (call1_df, call2_df) - DataFrames from first and second API calls
        """
        logger.info(f"=== ENHANCED EXTRACTION STARTED (BOTH CALLS) ===")
        logger.info(f"File: {file_path}")
        logger.info(f"Query: {query}")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Extract file base name for CSV logging
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Read file
        start_time = time.time()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                passage = f.read()
            read_time = time.time() - start_time
            logger.info(f"File read in {read_time:.2f}s ({len(passage)} chars)")
        except Exception as e:
            logger.error(f"File read failed: {e}")
            raise
        
        # Extract using OpenAI (both calls)
        try:
            raw1, raw2, prob_metrics = self.extract_with_openai(passage, query, file_base_name)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
        
        # Process Call 1 response
        logger.info("Processing Call 1 response...")
        try:
            data1 = self.clean_and_repair_json(raw1)
            if isinstance(data1, dict):
                flattened1 = self.flatten_nested_data(data1)
            elif isinstance(data1, list):
                flattened1 = data1
            else:
                raise ValueError(f"Unexpected data type from call 1: {type(data1)}")
            
            df1 = pd.DataFrame(flattened1)
            logger.info(f"Call 1 DataFrame created with {len(df1)} rows")
            
            # Add call 1 specific metrics
            if not df1.empty and prob_metrics:
                if 'avg_probability_1' in prob_metrics:
                    df1['model_probability'] = prob_metrics['avg_probability_1']
                    df1['confidence_level'] = df1['model_probability'].apply(
                        lambda x: 'HIGH' if x > 0.8 else 'MEDIUM' if x > 0.5 else 'LOW'
                    )
                    df1['needs_review'] = df1['model_probability'] < confidence_threshold
                
                for key in ['avg_logprob_1', 'token_count_1']:
                    if key in prob_metrics:
                        df1[f'model_{key}'] = prob_metrics[key]
                
                df1['call_type'] = 'initial_call'
            
        except Exception as e:
            logger.error(f"Call 1 processing failed: {e}")
            df1 = pd.DataFrame()  # Empty DataFrame on failure
        
        # Process Call 2 response  
        logger.info("Processing Call 2 response...")
        try:
            data2 = self.clean_and_repair_json(raw2)
            if isinstance(data2, dict):
                flattened2 = self.flatten_nested_data(data2)
            elif isinstance(data2, list):
                flattened2 = data2
            else:
                raise ValueError(f"Unexpected data type from call 2: {type(data2)}")
            
            df2 = pd.DataFrame(flattened2)
            logger.info(f"Call 2 DataFrame created with {len(df2)} rows")
            
            # Add call 2 specific metrics
            if not df2.empty and prob_metrics:
                if 'avg_probability_2' in prob_metrics:
                    df2['model_probability'] = prob_metrics['avg_probability_2']
                    df2['confidence_level'] = df2['model_probability'].apply(
                        lambda x: 'HIGH' if x > 0.8 else 'MEDIUM' if x > 0.5 else 'LOW'
                    )
                    df2['needs_review'] = df2['model_probability'] < confidence_threshold
                
                for key in ['avg_logprob_2', 'token_count_2']:
                    if key in prob_metrics:
                        df2[f'model_{key}'] = prob_metrics[key]
                
                df2['call_type'] = 'refinement_call'
                
        except Exception as e:
            logger.error(f"Call 2 processing failed: {e}")
            df2 = pd.DataFrame()  # Empty DataFrame on failure
        
        # Log final statistics
        total_time = time.time() - start_time
        logger.info(f"Call 1 DataFrame shape: {df1.shape}")
        logger.info(f"Call 2 DataFrame shape: {df2.shape}")
        
        # Log only essential summary information
        if not df1.empty and 'Statement' in df1.columns:
            logger.info(f"Call 1 - Unique statements: {df1['Statement'].nunique()}")
        if not df2.empty and 'Statement' in df2.columns:
            logger.info(f"Call 2 - Unique statements: {df2['Statement'].nunique()}")
        
        logger.info(f"=== BOTH CALLS EXTRACTION COMPLETE in {total_time:.2f}s ===\n")
        
        return df1, df2
    
    def extract_from_file(self, file_path: str, query: str, confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Extract financial data from file with enhanced error handling and probability tracking.
        
        Args:
            file_path: Path to the text file
            query: Extraction query
            confidence_threshold: Minimum confidence for automatic acceptance
            
        Returns:
            DataFrame with extracted financial data and probability columns
        """
        logger.info(f"=== ENHANCED EXTRACTION STARTED ===")
        logger.info(f"File: {file_path}")
        logger.info(f"Query: {query}")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Extract file base name for CSV logging
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Read file
        start_time = time.time()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                passage = f.read()
            read_time = time.time() - start_time
            logger.info(f"File read in {read_time:.2f}s ({len(passage)} chars)")
        except Exception as e:
            logger.error(f"File read failed: {e}")
            raise
        
        # Extract using OpenAI
        try:
            raw1, raw2, prob_metrics = self.extract_with_openai(passage, query, file_base_name)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
        
        # Clean and parse JSON
        try:
            data = self.clean_and_repair_json(raw2)
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            raise
        
        # Flatten and create DataFrame
        try:
            if isinstance(data, dict):
                flattened = self.flatten_nested_data(data)
            elif isinstance(data, list):
                flattened = data
            else:
                raise ValueError(f"Unexpected data type: {type(data)}")
            
            df = pd.DataFrame(flattened)
            logger.info(f"DataFrame created with {len(df)} rows")
            
            if df.empty:
                logger.warning("Empty DataFrame created")
                return df
            
            # Add probability columns if available
            if prob_metrics:
                for key, value in prob_metrics.items():
                    df[f'model_{key}'] = value
                    
                # Add confidence classification
                if 'avg_probability_2' in prob_metrics:
                    prob = prob_metrics['avg_probability_2']
                elif 'avg_probability_1' in prob_metrics:
                    prob = prob_metrics['avg_probability_1']
                else:
                    prob = None
                
                if prob is not None:
                    df['model_probability'] = prob
                    df['confidence_level'] = df['model_probability'].apply(
                        lambda x: 'HIGH' if x > 0.8 else 'MEDIUM' if x > 0.5 else 'LOW'
                    )
                    df['needs_review'] = df['model_probability'] < confidence_threshold
            
            # Log final statistics
            logger.info(f"Final DataFrame shape: {df.shape}")
            if 'Statement' in df.columns:
                logger.info(f"Unique statements: {df['Statement'].nunique()}")
            
            total_time = time.time() - start_time
            logger.info(f"=== EXTRACTION COMPLETE in {total_time:.2f}s ===\n")
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame creation failed: {e}")
            raise
    
    def flatten_nested_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested financial data structure."""
        logger.info("Flattening nested data structure...")
        flattened = []
        
        for statement_type, years_data in data.items():
            if not isinstance(years_data, dict):
                continue
                
            for year, items_data in years_data.items():
                if not isinstance(items_data, dict):
                    continue
                    
                for item_name, item_data in items_data.items():
                    if isinstance(item_data, dict):
                        row = {
                            'Statement': statement_type,
                            'Year': year,
                            'Item': item_name,
                            **item_data
                        }
                        flattened.append(row)
        
        logger.info(f"Flattened to {len(flattened)} rows")
        return flattened 