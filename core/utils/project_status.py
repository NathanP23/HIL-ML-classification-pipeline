"""
Project status and reporting utilities
"""
import os
import json
import pandas as pd
import glob
from config.paths import PATHS


def print_project_status():
	"""Print current status of the labeling project"""
	print("="*60)
	print("PROJECT STATUS")
	print("="*60)
	
	# Check labeled data - look for most recent master file
	master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
	master_files = glob.glob(master_pattern)
	labeled_file = sorted(master_files)[-1] if master_files else None
	
	if labeled_file and os.path.exists(labeled_file):
		with open(labeled_file, 'r', encoding='utf-8') as f:
			labeled_data = json.load(f)
		print(f"✅ Labeled examples: {len(labeled_data)} records")
	else:
		print("❌ No labeled data found")
		labeled_data = []
	
	# Check API predictions
	api_file = PATHS['api_predictions']
	if os.path.exists(api_file):
		with open(api_file, 'r', encoding='utf-8') as f:
			api_data = json.load(f)
		print(f"📊 API predictions: {len(api_data)} records")
	else:
		print("📊 No API predictions found")
	
	# Check consolidated data
	consolidated_file = PATHS['consolidated_data']
	if os.path.exists(consolidated_file):
		df = pd.read_excel(consolidated_file)
		print(f"📋 Total unique records: {len(df)}")
		
		if labeled_data:
			labeled_ids = {item['id'] for item in labeled_data}
			# Create hash IDs for comparison (simplified check)
			remaining_estimate = len(df) - len(labeled_ids)
			print(f"🔄 Estimated remaining: ~{remaining_estimate} records")
	else:
		print("📋 No consolidated data found")
	
	# Check TO_LABEL files
	outputs_dir = PATHS['manual_batches']
	to_label_files = [f for f in os.listdir(outputs_dir) if f.startswith('MANUAL_LABEL_') and '_AT-' in f] if os.path.exists(outputs_dir) else []
	if to_label_files:
		latest_file = sorted(to_label_files)[-1]
		print(f"⏳ Latest batch file: {latest_file}")
	
	print("="*60)


def export_progress_report():
	"""Export a progress report with statistics"""
	report = {
		"timestamp": pd.Timestamp.now().isoformat(),
		"files_status": {},
		"statistics": {}
	}
	
	# Check files
	files_to_check = [
		PATHS['api_predictions'],
		PATHS['consolidated_data'],
		PATHS['ft_data']
	]
	
	# Add most recent master file if it exists
	master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
	master_files = glob.glob(master_pattern)
	if master_files:
		files_to_check.append(sorted(master_files)[-1])
	
	for file in files_to_check:
		report["files_status"][file] = os.path.exists(file)
		if os.path.exists(file):
			if file.endswith('.json'):
				with open(file, 'r', encoding='utf-8') as f:
					data = json.load(f)
				report["statistics"][file] = {"record_count": len(data)}
			elif file.endswith('.xlsx'):
				df = pd.read_excel(file)
				report["statistics"][file] = {"record_count": len(df)}
	
	# Save report
	os.makedirs(PATHS['reports'], exist_ok=True)
	report_file = os.path.join(PATHS['reports'], f"progress_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
	with open(report_file, 'w', encoding='utf-8') as f:
		json.dump(report, f, ensure_ascii=False, indent=2)
	
	print(f"Progress report saved to: {report_file}")
	return report