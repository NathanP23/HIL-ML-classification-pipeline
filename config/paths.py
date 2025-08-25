"""
File paths configuration
"""
import os

# Base directories
BASE_OUTPUT_DIR = "outputs"
MANUAL_LABELING_DIR = os.path.join(BASE_OUTPUT_DIR, "manual_labeling")
FINE_TUNING_DIR = os.path.join(BASE_OUTPUT_DIR, "fine_tuning")

# Specific paths
PATHS = {
    # Base directory
    'base_output_dir': BASE_OUTPUT_DIR,
    # Data files
    'consolidated_data': os.path.join(BASE_OUTPUT_DIR, 'consolidated_data.xlsx'),
    
    # Manual labeling
    'manual_batches': os.path.join(MANUAL_LABELING_DIR, 'batches'),
    'manual_master': os.path.join(MANUAL_LABELING_DIR, 'master'),
    'api_predictions': os.path.join(MANUAL_LABELING_DIR, 'api_predictions.json'),
    
    # Fine-tuning
    'training_data': os.path.join(FINE_TUNING_DIR, 'training'),
    'ft_data': os.path.join(FINE_TUNING_DIR, 'training', 'ft_data.jsonl'),
    
    # Reports
    'reports': os.path.join(BASE_OUTPUT_DIR, 'reports'),
    
    # Anonymization
    'anonymized': 'anonymized'
}

def ensure_directories():
    """Create necessary directories if they don't exist"""
    for path in PATHS.values():
        if os.path.dirname(path):  # Only create if it's a directory path
            os.makedirs(os.path.dirname(path), exist_ok=True)