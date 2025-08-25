"""
Data anonymization functionality
"""
import re
import os
import pandas as pd


def anonymize_national_id(text):
    """Replace national ID numbers with sequential digits"""
    # Pattern for national ID (7 to 9 digits)
    pattern = r'\b(?<![\+\-\d])\d{7,9}\b(?![\-\d])'  # Added negative lookbehind/ahead
    pattern_with_dash = r'\b\d{6,8}[-]\d\b'
    
    def replace_with_sequence(match):
        num_length = len(match.group(0).replace('-', ''))
        return ''.join(str(i+1) for i in range(num_length))
    
    text = re.sub(pattern, replace_with_sequence, str(text))
    text = re.sub(pattern_with_dash, replace_with_sequence, text)
    return text


def anonymize_phone_numbers(text):
    """Replace phone numbers with standardized anonymous format"""
    patterns = [
        # International format
        (r'\+972[-]?\d[-]?\d{3}[-]?\d{4}', '+972555555555'),
        (r'\+9720[-]?\d[-]?\d{3}[-]?\d{4}', '+9720555555555'),
        
        # Mobile phones with different formats
        (r'05\d[-]?\d{3}[-]?\d{4}', '055-555-5555'),
        (r'05\d\d{7}', '0555555555'),
        (r'05\d[-]?\d{7}', '055-5555555'),
        
        # Landlines with different formats
        (r'0[2-9][-]?\d{3}[-]?\d{4}', '02-555-5555'),
        (r'0[2-9]\d{7}', '025555555'),
        (r'0[2-9][-]?\d{7}', '02-5555555'),
        
        # Special numbers
        (r'1[-]?[78]00[-]?\d{6}', '1-700-555555'),
        (r'1[78]00\d{6}', '1700555555')
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, str(text))
    
    return text


def anonymize_names_from_excel(text, excel_file="names/Custom.xlsx"):
    """Remove names from text using names from Excel file - OPTIMIZED"""
    if pd.isna(text):
        return text

    # Load names from Excel (with caching to avoid reloading each time)
    if not hasattr(anonymize_names_from_excel, '_names_cache'):
        try:
            df = pd.read_excel(excel_file)

            if 'names' not in df.columns:
                print(f"Warning: 'names' column not found in {excel_file}")
                anonymize_names_from_excel._names_cache = None
            else:
                # Create set of unique names
                names_set = set()
                for name in df['names'].dropna():
                    name = str(name).strip()
                    if name:
                        names_set.add(name)

                # Sort by length (longest first) and create single regex pattern
                sorted_names = sorted(names_set, key=len, reverse=True)

                # Create one big regex pattern for all names
                escaped_names = [re.escape(name) for name in sorted_names]
                combined_pattern = r'\b(?:' + '|'.join(escaped_names) + r')\b'

                anonymize_names_from_excel._names_cache = re.compile(combined_pattern, re.IGNORECASE)
                print(f"Loaded {len(names_set)} unique names from {excel_file}")

        except Exception as e:
            print(f"Error loading names from {excel_file}: {e}")
            anonymize_names_from_excel._names_cache = None

    # Apply name removal with single regex operation
    if anonymize_names_from_excel._names_cache:
        text = anonymize_names_from_excel._names_cache.sub('[פלוני אלמוני]', str(text))

    return text


def anonymize_data(dfs):
    """Apply anonymization to all DataFrames"""
    
    # Create anonymized folder if it doesn't exist
    anonymized_folder = "anonymized"
    if not os.path.exists(anonymized_folder):
        os.makedirs(anonymized_folder)
        print(f"Created folder: {anonymized_folder}")

    # Apply anonymization to all DataFrames
    for df_name, df in dfs.items():
        print(f"Anonymizing {df_name}...")
        # Apply to all string columns - updated to use text_content
        for col in ["text_content"]:
            if col in df.columns:
                df[col] = df[col].apply(anonymize_national_id)
                df[col] = df[col].apply(anonymize_phone_numbers)
                df[col] = df[col].apply(anonymize_names_from_excel)

        # Save to anonymized folder
        output_path = os.path.join(anonymized_folder, f"{df_name}_anonymized.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {df_name} to {output_path}")

    return dfs