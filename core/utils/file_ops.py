"""
File operations utilities
"""
import os
import json
import pandas as pd
from config.paths import PATHS


def cleanup_old_files(keep_latest=3):
    """Clean up old TO_LABEL files, keeping only the latest ones"""
    outputs_dir = PATHS['manual_batches']
    if not os.path.exists(outputs_dir):
        print("No manual labeling batches directory found")
        return
        
    to_label_files = [f for f in os.listdir(outputs_dir) if f.startswith('MANUAL_LABEL_') and '_AT-' in f]
    
    if len(to_label_files) <= keep_latest:
        print(f"Only {len(to_label_files)} TO_LABEL files found, nothing to clean up")
        return
    
    # Sort by filename (which includes timestamp)
    sorted_files = sorted(to_label_files)
    files_to_delete = sorted_files[:-keep_latest]  # Keep only the latest ones
    
    print(f"Cleaning up {len(files_to_delete)} old TO_LABEL files...")
    for file in files_to_delete:
        file_path = os.path.join(outputs_dir, file)
        os.remove(file_path)
        print(f"Deleted: {file}")
    
    print(f"Kept latest {keep_latest} files: {sorted_files[-keep_latest:]}")


def get_labeling_suggestions(df, labeled_ids, suggestion_type="diverse"):
    """Get suggestions for which records to label next"""
    unlabeled = df[~df["id"].isin(labeled_ids)]
    
    if len(unlabeled) == 0:
        print("All records have been labeled!")
        return pd.DataFrame()
    
    print(f"Found {len(unlabeled)} unlabeled records")
    
    if suggestion_type == "diverse":
        # Suggest records with diverse lengths
        lengths = unlabeled['text_content'].str.len()
        
        # Get records from different length quartiles
        q1, q2, q3 = lengths.quantile([0.25, 0.5, 0.75])
        
        short = unlabeled[lengths <= q1].head(3)
        medium = unlabeled[(lengths > q1) & (lengths <= q3)].head(3) 
        long = unlabeled[lengths > q3].head(4)
        
        suggestions = pd.concat([short, medium, long])
        
    elif suggestion_type == "shortest":
        suggestions = unlabeled.nsmallest(10, unlabeled['text_content'].str.len())
        
    else:  # longest (default)
        suggestions = unlabeled.nlargest(10, unlabeled['text_content'].str.len())
    
    print(f"Suggested {len(suggestions)} records for labeling ({suggestion_type} strategy)")
    return suggestions


def save_json(data, filepath, ensure_dir=True):
    """Save data to JSON file with optional directory creation"""
    if ensure_dir:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)