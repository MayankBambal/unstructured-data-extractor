#!/usr/bin/env python3
"""
Test script to validate all imports in the financial data extraction project.
Run this script to ensure all dependencies are properly installed.
"""

import sys
import traceback

def test_import(module_name, import_statement):
    """Test a single import and return result."""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: WARNING - {e}")
        return True

def main():
    """Test all required imports."""
    print("="*60)
    print("Testing all imports for Financial Data Extraction Project")
    print("="*60)
    
    tests = [
        # Core Python libraries
        ("os", "import os"),
        ("json", "import json"),
        ("re", "import re"),
        ("math", "import math"),
        ("time", "import time"),
        ("logging", "import logging"),
        ("concurrent.futures", "import concurrent.futures"),
        ("sys", "import sys"),
        ("shutil", "import shutil"),
        
        # Data processing
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        
        # Environment and configuration
        ("python-dotenv", "from dotenv import load_dotenv, find_dotenv"),
        
        # API clients (OpenAI only)
        ("openai", "from openai import OpenAI, AuthenticationError, OpenAIError"),
        
        # Web scraping and HTML processing
        ("requests", "import requests"),
        ("beautifulsoup4", "from bs4 import BeautifulSoup"),
        
        # SEC data downloading
        ("sec-edgar-downloader", "from sec_edgar_downloader import Downloader"),
        
        # File and content processing
        ("python-magic", "import magic"),
        
        # LangChain for advanced prompting
        ("langchain", "from langchain.output_parsers import PydanticOutputParser"),
        ("langchain", "from langchain.prompts import PromptTemplate"),
        ("pydantic", "import pydantic"),
        
        # Typing
        ("typing", "from typing import Any, Dict, List, Union, Tuple"),
    ]
    
    print("\nTesting imports...")
    print("-" * 40)
    
    passed = 0
    total = len(tests)
    
    for module_name, import_statement in tests:
        if test_import(module_name, import_statement):
            passed += 1
    
    print("-" * 40)
    print(f"Results: {passed}/{total} imports successful")
    
    if passed == total:
        print("🎉 All imports are working correctly!")
        return 0
    else:
        print("⚠️  Some imports failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 