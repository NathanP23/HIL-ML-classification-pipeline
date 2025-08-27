"""
Manual change detection from modified Excel files
"""
import json
import pandas as pd
import os
import glob
from config.settings import ALL_LABELS, get_prompt
from config.paths import PATHS


def _generate_jsonl_from_master(master_data):
	"""Generate JSONL file for fine-tuning from master data (same logic as label_manager.py)"""
	from core.labeling.prompt_builder import create_system_message_with_examples
	
	jsonl_file = PATHS['ft_data']
	os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
	
	with open(jsonl_file, "w", encoding="utf-8") as f:
		system_msg = create_system_message_with_examples()
		for item in master_data:
			messages = [
				{"role": "system", "content": system_msg.strip()},
				{"role": "user", "content": get_prompt("user_with_keys", text=item['text_content'], keys=ALL_LABELS)},
				{"role": "assistant", "content": json.dumps({k: int(item[k]) for k in ALL_LABELS}, ensure_ascii=False)}
			]
			f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
	
	print(f"   📁 Updated JSONL file: {jsonl_file} ({len(master_data)} examples)")


def detect_changes_from_excel(original_json_path, excel_path):
	"""
	Compare original JSON with modified Excel and extract changes
	
	Args:
		original_json_path: Path to original JSON file
		excel_path: Path to modified Excel file
	
	Returns:
		List of changed records
	"""
	print(f"📂 Loading original: {os.path.basename(original_json_path)}")
	with open(original_json_path, 'r', encoding='utf-8') as f:
		original_data = json.load(f)
	
	original_lookup = {item['id']: item for item in original_data}
	
	print(f"📊 Loading modified: {os.path.basename(excel_path)}")
	modified_df = pd.read_excel(excel_path, sheet_name='Classifications')
	
	print("🔍 Detecting changes...")
	changes = []
	
	for _, row in modified_df.iterrows():
		record_id = str(row['id'])
		
		if record_id not in original_lookup:
			continue
		
		original_record = original_lookup[record_id]
		has_changes = False
		
		# Create updated record
		updated_record = {
			'id': record_id,
			'text_content': original_record['text_content']
		}
		
		# Check each category for changes
		for category in ALL_LABELS:
			original_value = original_record.get(category, 0)
			modified_value = int(row.get(category, 0)) if pd.notna(row.get(category, 0)) else 0
			
			updated_record[category] = modified_value
			
			if original_value != modified_value:
				has_changes = True
		
		if has_changes:
			changes.append(updated_record)
	
	return changes


def save_manual_changes(changes, output_path=None):
	"""Save manual changes to JSON file in proper manual labeling batch format"""
	if output_path is None:
		timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
		sample_size = len(changes)
		
		# Use proper manual labeling batch directory and filename format
		output_dir = PATHS['manual_batches']
		os.makedirs(output_dir, exist_ok=True)
		
		filename = f"MANUAL_LABEL_EXCEL_AT-{timestamp}_SAMPLE_SIZE_{sample_size}.json"
		output_path = os.path.join(output_dir, filename)
	
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(changes, f, ensure_ascii=False, indent=2)
	
	return output_path


def show_change_summary(changes):
	"""Generate summary of changes"""
	if not changes:
		print("✅ No changes detected!")
		return
	
	print(f"\n📊 MANUAL CHANGES SUMMARY:")
	print(f"   Total modified records: {len(changes)}")
	
	# Show sample changes
	print(f"\n🔍 SAMPLE CHANGED RECORDS:")
	for i, change in enumerate(changes[:3]):  # Show first 3 changes
		print(f"\n   Record {i+1} - ID: {change['id']}")
		print(f"   Text: {change['text_content'][:80]}...")
		
		# Count active labels
		active_labels = [cat for cat in ALL_LABELS if change.get(cat, 0) == 1]
		print(f"   Active labels ({len(active_labels)}): {', '.join(active_labels[:5])}{'...' if len(active_labels) > 5 else ''}")


def integrate_changes_with_master(changes):
	"""Integrate changes with existing manual labeling system"""
	print("\n🔗 INTEGRATING WITH MANUAL LABELS SYSTEM...")
	
	try:
		# Find latest manual label file
		master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
		existing_files = glob.glob(master_pattern)
		
		if existing_files:
			latest_file = sorted(existing_files)[-1]
			
			# Load existing master
			with open(latest_file, 'r', encoding='utf-8') as f:
				master_data = json.load(f)
			
			# Create lookup
			master_lookup = {item['id']: item for item in master_data}
			
			# Update with changes and add new records
			updated_count = 0
			added_count = 0
			
			for change in changes:
				record_id = change['id']
				
				if record_id in master_lookup:
					# Update existing record
					for i, item in enumerate(master_data):
						if item['id'] == record_id:
							master_data[i] = change
							updated_count += 1
							break
				else:
					# Add new record
					master_data.append(change)
					added_count += 1
			
			# Save updated master file
			timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
			total_samples = len(master_data)
			new_master_file = os.path.join(PATHS['manual_master'], 
										 f"TOTAL_MANUAL_LABEL_AT-{timestamp}_TOTAL_SAMPLE_SIZE_{total_samples}.json")
			
			os.makedirs(os.path.dirname(new_master_file), exist_ok=True)
			
			with open(new_master_file, 'w', encoding='utf-8') as f:
				json.dump(master_data, f, ensure_ascii=False, indent=2)
			
			# Remove old master file
			if existing_files:
				os.remove(latest_file)
			
			# Generate JSONL for fine-tuning (same as label_manager.py)
			_generate_jsonl_from_master(master_data)
			
			print(f"   ✅ Updated master file: {os.path.basename(new_master_file)}")
			print(f"   📊 Updated records: {updated_count}")
			print(f"   📊 Added records: {added_count}")
			print(f"   📊 Total manual labels: {total_samples}")
			
			return new_master_file
			
		else:
			print("   ⚠️  No existing master file found. Creating new one...")
			
			# Create new master file with just the changes
			timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
			new_master_file = os.path.join(PATHS['manual_master'], 
										 f"TOTAL_MANUAL_LABEL_AT-{timestamp}_TOTAL_SAMPLE_SIZE_{len(changes)}.json")
			
			os.makedirs(os.path.dirname(new_master_file), exist_ok=True)
			
			with open(new_master_file, 'w', encoding='utf-8') as f:
				json.dump(changes, f, ensure_ascii=False, indent=2)
			
			# Generate JSONL for fine-tuning
			_generate_jsonl_from_master(changes)
			
			print(f"   ✅ Created master file: {os.path.basename(new_master_file)}")
			print(f"   📊 Total manual labels: {len(changes)}")
			
			return new_master_file
	
	except Exception as e:
		print(f"   ❌ Integration error: {e}")
		return None


def process_excel_changes(original_json_path, excel_path, integrate=True):
	"""
	Complete workflow to process Excel changes
	
	Args:
		original_json_path: Path to original JSON file
		excel_path: Path to modified Excel file
		integrate: Whether to integrate with master labels
	
	Returns:
		Tuple of (changes_json_path, master_file_path)
	"""
	try:
		# Detect changes
		changes = detect_changes_from_excel(original_json_path, excel_path)
		
		# Show summary
		show_change_summary(changes)
		
		if not changes:
			return None, None
		
		# Save standalone changes
		changes_json_path = save_manual_changes(changes)
		
		# Show full absolute path
		absolute_changes_path = os.path.abspath(changes_json_path)
		print(f"\n💾 Changes saved to: {absolute_changes_path}")
		
		# Integrate with master if needed
		master_file_path = None
		if integrate:
			master_file_path = integrate_changes_with_master(changes)
		
		print(f"\n🎉 EXCEL CHANGES PROCESSED SUCCESSFULLY!")
		print(f"   📊 Modified records: {len(changes)}")
		print(f"   📁 Changes JSON: {absolute_changes_path}")
		if master_file_path:
			print(f"   📁 Updated master: {os.path.basename(master_file_path)}")
		
		return changes_json_path, master_file_path
		
	except Exception as e:
		print(f"❌ Error processing Excel changes: {e}")
		return None, None