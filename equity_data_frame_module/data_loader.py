"""
Data loading utilities for Bloomberg analysis.
"""

import pandas as pd
import numpy as np
import os

class DataLoader:
    """Handles loading and basic processing of Bloomberg Excel data."""
    
    def __init__(self, display_options=True):
        if display_options:
            self.setup_pandas_display()
    
    def setup_pandas_display(self):
        """Configure pandas display options for better data viewing."""
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', '{:.4f}'.format)
    
    def read_excel_file(self, file_path, sheet_name=None):
        """
        Read an Excel file using pandas with comprehensive error handling.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file
        sheet_name : str or None
            Name of the sheet to read. If None, reads the first sheet
        
        Returns:
        --------
        pandas.DataFrame : The loaded data
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"Successfully loaded sheet '{sheet_name}' from {file_path}")
            else:
                df = pd.read_excel(file_path)
                print(f"Successfully loaded first sheet from {file_path}")
            
            # Display basic information
            print(f"DataFrame Shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            return df
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
    
    def read_all_sheets(self, file_path):
        """
        Read all sheets from Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file
            
        Returns:
        --------
        dict : Dictionary with sheet names as keys and DataFrames as values
        """
        try:
            # Read all sheets into a dictionary
            all_sheets = pd.read_excel(file_path, sheet_name=None)
            
            print(f"Found {len(all_sheets)} sheets:")
            for sheet_name, df in all_sheets.items():
                print(f"  - {sheet_name}: {df.shape}")
            
            return all_sheets
        except Exception as e:
            print(f"Error reading sheets: {e}")
            return None
    
    def create_synthetic_data(self, n_stocks=100, seed=42):
        """
        Create synthetic Bloomberg-style equity data for demonstration.
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to generate
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        pandas.DataFrame : Synthetic Bloomberg-style dataset
        """
        np.random.seed(seed)
        
        df = pd.DataFrame({
            'Ticker': [f'STK{i:03d}' for i in range(1, n_stocks+1)],
            'Company Name': [f'Company {i}' for i in range(1, n_stocks+1)],
            'Sector': np.random.choice(['Technology', 'Financials', 'Healthcare', 'Energy', 
                                       'Industrials', 'Consumer Staples', 'Utilities'], n_stocks),
            'Country': np.random.choice(['USA', 'UK', 'Germany', 'Japan', 'Canada', 'France'], n_stocks),
            'Price (USD)': np.random.uniform(50, 1000, n_stocks),
            'Market Cap (USD Billion)': np.random.lognormal(5, 1.5, n_stocks),
            'PE Ratio (TTM)': np.random.lognormal(3, 0.5, n_stocks),
            'Forward PE': np.random.lognormal(2.8, 0.4, n_stocks),
            'Price/Sales (TTM)': np.random.lognormal(1.5, 0.8, n_stocks),
            'EV/EBITDA (TTM)': np.random.lognormal(2.5, 0.6, n_stocks),
            'Price/Book (TTM)': np.random.lognormal(1, 0.7, n_stocks),
            'ROE (%)': np.random.normal(15, 10, n_stocks),
            'ROA (%)': np.random.normal(8, 5, n_stocks),
            'Net Margin (%)': np.random.normal(10, 8, n_stocks),
            'Revenue Growth YoY (%)': np.random.normal(8, 15, n_stocks),
            'EPS Growth YoY (%)': np.random.normal(12, 25, n_stocks),
            'Debt/Equity': np.random.lognormal(0, 1, n_stocks),
            'Current Ratio': np.random.lognormal(0.5, 0.4, n_stocks),
            '1W % Change': np.random.normal(0, 3, n_stocks),
            '1M % Change': np.random.normal(0, 8, n_stocks),
        })
        
        print(f"✓ Synthetic dataset created: {df.shape[0]} stocks, {df.shape[1]} features")
        return df

def read_excel_advanced(file_path, **kwargs):
    """
    Advanced Excel reading with multiple options.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    **kwargs : dict
        Additional arguments for pd.read_excel()
        Common options:
        - sheet_name: str, int, list, or None (default 0)
        - header: int, list of int, default 0
        - skiprows: list-like, int, or callable, optional
        - usecols: str, list-like, or callable, optional
        - nrows: int, optional
        - index_col: int, str, sequence, or False, default None
    
    Returns:
    --------
    pandas.DataFrame or None : Loaded data or None if error
    """
    try:
        df = pd.read_excel(file_path, **kwargs)
        print(f"Loaded Excel file with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None