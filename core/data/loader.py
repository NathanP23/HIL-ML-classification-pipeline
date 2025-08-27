"""
Data loading and consolidation functionality
"""
import pandas as pd
from config.settings import COLUMN_MAPPINGS, FILES



def load_data():
	"""Load data from Excel files with column mapping"""
	print("Loading data from Excel files:")
	
	# Create a reverse mapping dictionary (Original to Standard)
	reverse_mapping = {}
	for eng, mixed_list in COLUMN_MAPPINGS.items():
		for name in mixed_list:
			reverse_mapping[name] = eng
	print("Column mappings created")
	
	# Read all files and apply mappings
	dfs = {}
	for i, (file_key, file_info) in enumerate(FILES.items(), start=1):
		# Read the Excel file with the specified sheet
		df = pd.read_excel(
			file_key,
			sheet_name=file_info['sheet_name']
		)
		print("Data loaded from {file_key}")
		
		# Rename columns where mapping exists
		df = df.rename(columns=reverse_mapping)
		
		# Store in dictionary with auto numbering
		dfs[f"df{i}"] = df
	
	# Print success message 
	print("Data loaded successfully")
	return dfs



def consolidate_data(all_df):
	"""Consolidate data from multiple dataframes"""
	print("Consolidating data...")
	
	all_records = []

	for df_name, df in all_df.items():
		if 'text_content' in df.columns:
			valid_records = df['text_content']
			valid_records = valid_records[
				valid_records.notna() &
				(valid_records.str.strip() != '') &
				(valid_records.str.lower() != 'nan')
			]

			cleaned_df = pd.DataFrame({
				'text_content': valid_records,
				'source_df': df_name
			})

			all_records.append(cleaned_df)

	# Combine everything
	combined_df = pd.concat(all_records, ignore_index=True)

	# Group and summarize
	grouped = combined_df.groupby('text_content').agg(
		appearance_count=('text_content', 'count'),
		source_dfs=('source_df', lambda x: sorted(set(x)))
	).reset_index()

	# Print stats
	total_valid = len(combined_df)
	total_unique = len(grouped)
	appear_more_than_once = (grouped['appearance_count'] > 1).sum()
	appear_once = (grouped['appearance_count'] == 1).sum()

	print("=== Summary Stats ===")
	print("Total valid records: {total_valid}")
	print("Unique records: {total_unique}")
	print("Records appearing more than once: {appear_more_than_once}")
	print("Records appearing exactly once: {appear_once}")

	# Print success message 
	print("Data consolidated successfully. DataFrame shape: {grouped.shape}")
	return grouped