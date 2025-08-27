"""
Menu handlers extracted from main.py
"""
import os
from openai import OpenAI
from core.data.loader import load_data, consolidate_data
from core.data.anonymizer import anonymize_data
from core.labeling.label_manager import prepare_for_labeling, update_master_labels
from core.labeling.batch_processor import process_batch_for_labeling
from core.models.evaluation import (
	test_api_performance_baseline, 
	test_api_performance_leave_one_out, 
	compare_api_vs_manual_corrections, 
	test_fine_tuned_model
)
from core.models.fine_tuning import (
	upload_training_file, 
	create_fine_tune_job, 
	check_fine_tune_status, 
	list_fine_tune_jobs, 
	estimate_fine_tuning_cost
)
from core.models.bulk_classifier import run_bulk_classification
from core.utils.project_status import print_project_status, export_progress_report
from core.utils.file_ops import cleanup_old_files
from core.utils.excel_export import (
	convert_json_to_excel_rtl,
	export_latest_bulk_results,
	export_manual_labels
)
from core.utils.change_detection import process_excel_changes
from config.paths import PATHS


def setup_openai_client():
	"""Initialize OpenAI client"""
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("OPENAI_API_KEY environment variable is not set")
	return OpenAI(api_key=api_key)



def load_from_anonymized_csv():
	"""Load data from already anonymized CSV files"""
	import pandas as pd
	
	prefix = get_tree_prefix()
	
	anonymized_dir = PATHS['anonymized']
	if not os.path.exists(anonymized_dir):
		raise FileNotFoundError(f"Anonymized directory not found: {anonymized_dir}")
	
	# Expected CSV files
	csv_files = ['df1_anonymized.csv', 'df2_anonymized.csv', 'df3_anonymized.csv']
	all_df = {}
	
	for i, csv_file in enumerate(csv_files, 1):
		csv_path = os.path.join(anonymized_dir, csv_file)
		if os.path.exists(csv_path):
			all_df[f'df{i}'] = pd.read_csv(csv_path)
			print(f"{prefix}Loaded {csv_file}: {len(all_df[f'df{i}'])} rows")
		else:
			print(f"{prefix}Warning: {csv_file} not found, skipping...")
	
	if not all_df:
		raise FileNotFoundError("No anonymized CSV files found in the anonymized directory")
	
	return all_df



def run_data_preparation():
	"""Run data preparation pipeline"""
	
	print("\n" + "=" * 50)
	print("DATA PREPARATION OPTIONS")
	print("=" * 50)
	print("1. Load from raw Excel files (requires original data)")
	print("2. Load from anonymized CSV files (uses already processed data)")
	print("0. Back to main menu")
	print("=" * 50)
	
	sub_choice = input("Enter your choice (0-2): ").strip()
	
	if sub_choice == "0":
		return None
	elif sub_choice == "1":
		prefix = get_tree_prefix()
		print(f"{prefix}Data Preparation Pipeline - From Raw Excel")
		# Step 1: Load data from Excel
		all_df = load_data()

		# Step 2: Anonymize data
		all_df = anonymize_data(all_df)
		
	elif sub_choice == "2":
		prefix = get_tree_prefix()
		print(f"{prefix}Data Preparation Pipeline - From Anonymized CSV")
		# Load from already anonymized CSV files
		all_df = load_from_anonymized_csv()
		print(f"{prefix}Anonymized data loaded successfully")
		
	else:
		print("Invalid choice")
		return None

	# Step 3: Consolidate data (common for both paths)
	consolidated_df = consolidate_data(all_df)

	return consolidated_df


def run_single_labeling_batch(df=None, batch_size=10, selection_method="longest", max_examples=30):
	"""Run a single labeling batch (steps 4-7)"""
	print("=" * 60)
	print("STEP 4-7: SINGLE LABELING BATCH")
	print("=" * 60)
	
	if df is None:
		print("No DataFrame provided. Run data preparation first.")
		return None
	
	# Step 4: Prepare for labeling
	print("Step 4: Preparing data for labeling...")
	df, labeled_ids = prepare_for_labeling(df)
	
	# Steps 5-7: Process batch
	print(f"\nSteps 5-7: Processing batch of {batch_size} records...")
	client = setup_openai_client()
	
	to_label_file, records = process_batch_for_labeling(
		df, labeled_ids, client, batch_size, selection_method, max_examples
	)
	
	if to_label_file:
		print(f"\n✅ Batch ready for manual correction: {to_label_file}")
		print("📝 Next steps:")
		print("   1. Manually correct the predictions in the TO_LABEL file")
		print("   2. Run update_master_labels() to save corrections")
		print("   3. Repeat for next batch")
	
	return to_label_file


def print_manual_prompt():
	"""Print manual labeling prompt for ChatGPT"""
	# This functionality would be moved to a separate module if needed
	# For now, keeping it simple
	print("Manual labeling prompt functionality - implement if needed")

def run_fine_tuning_menu(consolidated_df=None):
	"""Complete fine-tuning operations submenu"""
	client = setup_openai_client()
	
	while True:
		print("\n" + "=" * 60)
		print("FINE-TUNING OPERATIONS")
		print("=" * 60)
		print("1. Estimate fine-tuning cost")
		print("2. Upload training data and start fine-tuning")
		print("3. Check fine-tuning job status")
		print("4. List all fine-tuning jobs")
		print("5. Test fine-tuned model against manual labels")
		print("6. Bulk classify all unlabeled data")
		print("7. Compare model performance (baseline/few-shot/fine-tuned)")
		print("0. Back to main menu")
		print("=" * 60)
		
		choice = input("Enter your choice (0-7): ").strip()
		
		if choice == "1":
			try:
				estimate_fine_tuning_cost()
			except Exception as e:
				print(f"❌ Error estimating cost: {e}")
		
		elif choice == "2":
			try:
				# Get hyperparameters
				print("📋 Configure hyperparameters:")
				
				epochs = int(input("Epochs (recommended: 3-4, default 3): ").strip() or "3")
				batch_size = int(input("Batch size (recommended: 1, default 1): ").strip() or "1") 
				lr_mult = float(input("Learning rate multiplier (recommended: 1.0-1.2, default 1.0): ").strip() or "1.0")
				suffix = input("Model suffix (optional, press Enter to skip): ").strip() or None
				
				print("📤 Uploading training data...")
				training_file_id = upload_training_file(client)
				
				print("🚀 Starting fine-tuning job...")
				job = create_fine_tune_job(client, training_file_id, suffix=suffix,
										 epochs=epochs, batch_size=batch_size, 
										 learning_rate_multiplier=lr_mult)
				
				print(f"✅ Fine-tuning started! Job ID: {job.id}")
				print("💡 Use option 3 to check status")
				
			except Exception as e:
				print(f"❌ Error starting fine-tuning: {e}")
		
		elif choice == "3":
			try:
				job_id = input("Enter job ID: ").strip()
				model = check_fine_tune_status(client, job_id)
				if model:
					print(f"🎉 Model ready for use: {model}")
			except Exception as e:
				print(f"❌ Error checking status: {e}")
		
		elif choice == "4":
			try:
				list_fine_tune_jobs(client)
			except Exception as e:
				print(f"❌ Error listing jobs: {e}")
		
		elif choice == "5":
			try:
				model_name = input("Enter fine-tuned model name: ").strip()
				print("Testing fine-tuned model against your manual labels...")
				ft_stats, ft_acc = test_fine_tuned_model(model_name, client=client)
				print(f"Fine-tuned model accuracy: {ft_acc:.1f}%")
			except Exception as e:
				print(f"❌ Error testing model: {e}")
		
		elif choice == "6":
			try:
				if consolidated_df is None:
					print("❌ No DataFrame available. Run data preparation first.")
					continue
				
				print("\n📋 BULK CLASSIFICATION OPTIONS:")
				print("1. Classify unlabeled data only (standard)")
				print("2. Classify ALL data including manual labels")
				
				bulk_choice = input("Enter choice (1-2, default 1): ").strip()
				include_manual = bulk_choice == "2"
				
				if include_manual:
					print("\n⚠️  WARNING: This will re-classify your manual labels!")
					print("   Use this to compare model performance or test new models.")
					confirm = input("Continue? (y/N): ").strip().lower()
					if confirm != 'y':
						print("Bulk classification cancelled.")
						continue
				
				model_name = input("Enter fine-tuned model name: ").strip()
				batch_size = int(input("Batch size for processing (default 50): ").strip() or "50")
				
				result = run_bulk_classification(client, model_name, consolidated_df, batch_size, include_manual=include_manual)
				if result:
					final_file, predictions, source_tracking = result
					print(f"✅ Bulk classification completed: {len(predictions)} records classified")
					
					if include_manual and source_tracking:
						manual_count = sum(1 for s in source_tracking.values() if s == 'previously_manual')
						print(f"   🔄 Re-classified {manual_count} previously manual labels")
			except Exception as e:
				print(f"❌ Error in bulk classification: {e}")
		
		elif choice == "7":
			try:
				print("\n" + "=" * 60)
				print("MODEL PERFORMANCE COMPARISON")
				print("=" * 60)
				print("1. Test baseline performance (no examples)")
				print("2. Test leave-one-out validation (few-shot)")
				print("3. Compare API predictions vs manual corrections")
				print("0. Back to fine-tuning menu")
				
				eval_choice = input("Enter your choice (0-3): ").strip()
				
				if eval_choice == "1":
					baseline_stats, baseline_acc = test_api_performance_baseline(client=client)
					print(f"Baseline accuracy: {baseline_acc:.1f}%")
				elif eval_choice == "2":
					loo_stats, loo_acc = test_api_performance_leave_one_out(client=client)
					print(f"Leave-one-out accuracy: {loo_acc:.1f}%")
				elif eval_choice == "3":
					comparison_stats, comparison_acc = compare_api_vs_manual_corrections()
					print(f"API prediction accuracy: {comparison_acc:.1f}%")
				
			except Exception as e:
				print(f"❌ Error in performance comparison: {e}")
		
		elif choice == "0":
			break
		
		else:
			print("Invalid choice. Please try again.")


def run_excel_export_menu():
	"""Excel export and manual editing menu"""
	while True:
		print("\n" + "=" * 60)
		print("EXCEL EXPORT & MANUAL EDITING")
		print("=" * 60)
		print("1. Export latest bulk results to Excel")
		print("2. Export bulk results + manual labels combined")
		print("3. Export manual labels only to Excel")
		print("4. Export custom JSON file to Excel")
		print("5. Process Excel changes back to JSON")
		print("0. Back to main menu")
		print("=" * 60)
		
		choice = input("Enter your choice (0-5): ").strip()
		
		if choice == "1":
			try:
				sort_choice = input("Sort by text length? (Y/n): ").strip().lower()
				sort_by_length = sort_choice != 'n'
				
				length_col_choice = input("Include text length column? (y/N): ").strip().lower()
				include_length_column = length_col_choice == 'y'
				
				excel_path = export_latest_bulk_results(include_manual_labels=False, sort_by_length=sort_by_length, include_length_column=include_length_column)
				if excel_path:
					print(f"✅ Bulk results exported to Excel")
					print(f"💡 You can now edit values manually in Excel")
			except Exception as e:
				print(f"❌ Error exporting bulk results: {e}")
		
		elif choice == "2":
			try:
				sort_choice = input("Sort by text length? (Y/n): ").strip().lower()
				sort_by_length = sort_choice != 'n'
				
				length_col_choice = input("Include text length column? (y/N): ").strip().lower()
				include_length_column = length_col_choice == 'y'
				
				excel_path = export_latest_bulk_results(include_manual_labels=True, sort_by_length=sort_by_length, include_length_column=include_length_column)
				if excel_path:
					print(f"✅ Combined bulk + manual results exported to Excel")
					print(f"💡 Source column shows manual vs model classifications")
					print(f"💡 You can now edit values manually in Excel")
			except Exception as e:
				print(f"❌ Error exporting combined results: {e}")
		
		elif choice == "3":
			try:
				sort_choice = input("Sort by text length? (Y/n): ").strip().lower()
				sort_by_length = sort_choice != 'n'
				
				length_col_choice = input("Include text length column? (y/N): ").strip().lower()
				include_length_column = length_col_choice == 'y'
				
				excel_path = export_manual_labels(sort_by_length=sort_by_length, include_length_column=include_length_column)
				if excel_path:
					print(f"✅ Manual labels exported to Excel")
					print(f"💡 You can now edit values manually in Excel")
			except Exception as e:
				print(f"❌ Error exporting manual labels: {e}")
		
		elif choice == "4":
			try:
				json_path = input("Enter path to JSON file: ").strip()
				if json_path and os.path.exists(json_path):
					output_path = input("Enter output Excel path (or press Enter for auto): ").strip()
					output_path = output_path if output_path else None
					
					include_manual = input("Include manual labels? (y/N): ").strip().lower() == 'y'
					sort_choice = input("Sort by text length? (Y/n): ").strip().lower()
					sort_by_length = sort_choice != 'n'
					length_col_choice = input("Include text length column? (y/N): ").strip().lower()
					include_length_column = length_col_choice == 'y'
					
					excel_path = convert_json_to_excel_rtl(json_path, output_path, include_manual=include_manual, sort_by_length=sort_by_length, include_length_column=include_length_column)
					print(f"✅ JSON exported to Excel: {excel_path}")
					print(f"💡 Features: RTL layout, freeze panes, source tracking")
				else:
					print("❌ Invalid or missing JSON file path")
			except Exception as e:
				print(f"❌ Error exporting JSON: {e}")
		
		elif choice == "5":
			try:
				print("\n📋 PROCESS EXCEL CHANGES")
				original_json = input("Enter path to original JSON file: ").strip()
				excel_path = input("Enter path to modified Excel file: ").strip()
				
				if original_json and excel_path and os.path.exists(original_json) and os.path.exists(excel_path):
					integrate = input("Integrate changes with manual labels system? (Y/n): ").strip().lower()
					integrate = integrate != 'n'
					
					changes_json, master_file = process_excel_changes(original_json, excel_path, integrate)
					
					if changes_json:
						print(f"✅ Excel changes processed successfully")
				else:
					print("❌ Invalid file paths provided")
			except Exception as e:
				print(f"❌ Error processing Excel changes: {e}")
		
		elif choice == "0":
			break
		
		else:
			print("Invalid choice. Please try again.")


def run_project_utilities_menu():
	"""Project utilities submenu"""
	while True:
		print("\n" + "=" * 60)
		print("PROJECT UTILITIES")
		print("=" * 60)
		print("1. Print project status")
		print("2. Cleanup old files")
		print("3. Export progress report")
		print("0. Back to main menu")
		print("=" * 60)
		
		choice = input("Enter your choice (0-3): ").strip()
		
		if choice == "1":
			try:
				print_project_status()
			except Exception as e:
				print(f"❌ Error printing status: {e}")
		
		elif choice == "2":
			try:
				keep = int(input("How many recent TO_LABEL files to keep? (default 3): ") or "3")
				cleanup_old_files(keep)
			except Exception as e:
				print(f"❌ Error in cleanup: {e}")
		
		elif choice == "3":
			try:
				report = export_progress_report()
				print("✅ Progress report exported")
			except Exception as e:
				print(f"❌ Error exporting report: {e}")
		
		elif choice == "0":
			break
		
		else:
			print("Invalid choice. Please try again.")