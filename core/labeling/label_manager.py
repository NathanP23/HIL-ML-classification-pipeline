"""
Label management functionality - ID creation, master label updates
"""
import os
import json
import hashlib
import glob
import pandas as pd
from config.settings import ALL_LABELS, get_prompt
from config.paths import PATHS
from core.utils.file_ops import save_json
from .prompt_builder import create_system_message_with_examples


def create_stable_ids(df):
	"""Create stable content-based hash IDs for DataFrame"""
	df['id'] = df['text_content'].apply(
		lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[:8]
	)
	return df


def load_labeled_ids(labeled_json_file=None):
	"""Load already labeled IDs from JSON file"""
	labeled_ids = set()
	
	# Find the most recent TOTAL_MANUAL_LABEL file if no specific file provided
	if labeled_json_file is None:
		master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
		existing_master_files = glob.glob(master_pattern)
		if existing_master_files:
			labeled_json_file = sorted(existing_master_files)[-1]  # Use most recent
		else:
			return labeled_ids  # No labeled files exist yet
	
	if os.path.exists(labeled_json_file):
		with open(labeled_json_file, 'r', encoding='utf-8') as f:
			master_data = json.load(f)
		labeled_ids = {item['id'] for item in master_data}
	
	return labeled_ids


def save_api_predictions(records):
	"""Save API predictions immediately (before manual correction)"""
	api_predictions_file = PATHS['api_predictions']
	os.makedirs(os.path.dirname(api_predictions_file), exist_ok=True)
	
	api_predictions = []
	if os.path.exists(api_predictions_file):
		with open(api_predictions_file, 'r', encoding='utf-8') as f:
			api_predictions = json.load(f)

	api_predictions.extend(records)
	save_json(api_predictions, api_predictions_file, ensure_dir=False)
	
	print(f"API predictions saved to {api_predictions_file} ({len(api_predictions)} total)")


def update_master_labels(last_labeled_json, master_json_file=None):
	"""Update master labeled examples and generate JSONL for potential fine-tuning"""
	# Load newly corrected JSON
	with open(last_labeled_json, 'r', encoding='utf-8') as f:
		new = json.load(f)

	# Find existing master file or create new one
	master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
	existing_master_files = glob.glob(master_pattern)
	
	# Merge with existing master if it exists
	if existing_master_files:
		# Use the most recent master file
		latest_master_file = sorted(existing_master_files)[-1]
		with open(latest_master_file, 'r', encoding='utf-8') as f:
			master = json.load(f)
		# Append and dedupe
		master += new
		unique = {item['id']: item for item in master}
		master = list(unique.values())
	else:
		master = new

	# Create new filename with timestamp and total sample size
	import time
	timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
	total_samples = len(master)
	
	# Ensure unique filename by adding microseconds if needed
	base_filename = f"TOTAL_MANUAL_LABEL_AT-{timestamp}_TOTAL_SAMPLE_SIZE_{total_samples}.json"
	master_json_file = os.path.join(PATHS['manual_master'], base_filename)
	
	# If file already exists, add microseconds to make it unique
	if os.path.exists(master_json_file):
		timestamp_with_microseconds = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')
		base_filename = f"TOTAL_MANUAL_LABEL_AT-{timestamp_with_microseconds}_TOTAL_SAMPLE_SIZE_{total_samples}.json"
		master_json_file = os.path.join(PATHS['manual_master'], base_filename)

	# Save updated master JSON with new filename
	save_json(master, master_json_file, ensure_dir=True)
	
	# Remove old master files to keep only the latest
	for old_file in existing_master_files:
		os.remove(old_file)

	print(f"✅ Master labeled examples updated: {len(master)} items.")
	print(f"💡 Next batch will use these {len(master)} examples for improved few-shot learning.")

	# Generate JSONL for potential future fine-tuning
	jsonl_file = PATHS['ft_data']
	os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
	
	with open(jsonl_file, "w", encoding="utf-8") as f:
		system_msg = create_system_message_with_examples()
		for item in master:
			messages = [
				{"role": "system", "content": system_msg.strip()},
				{"role": "user", "content": get_prompt("user_with_keys", text=item['text_content'], keys=ALL_LABELS)},
				{"role": "assistant", "content": json.dumps({k: int(item[k]) for k in ALL_LABELS}, ensure_ascii=False)}
			]
			f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

	print(f"📁 Also saved {len(master)} examples to `{jsonl_file}` for potential future fine-tuning.")
	print("🚀 No automatic fine-tuning triggered - cost savings achieved!")

	return master


def prepare_for_labeling(all_df, labeled_json_file=None):
	"""Prepare DataFrame with stable IDs and track already labeled ones"""
	# Create stable IDs based on content hash
	all_df = create_stable_ids(all_df)
	
	# Track already labeled IDs
	labeled_ids = load_labeled_ids(labeled_json_file)
	
	print(f"Found labeled IDs: {len(labeled_ids)}")
	print(f"DataFrame shape: {all_df.shape}")
	print(f"Unique IDs in DataFrame: {all_df['id'].nunique()}")
	print(f"Sample hash IDs: {all_df['id'].head().tolist()}")
	
	return all_df, labeled_ids