"""
Main orchestration script for the text classification project - OPTIMIZED VERSION
"""
import os
import sys
from dotenv import load_dotenv

# Prevent creation of __pycache__ folders
sys.dont_write_bytecode = True

# Add project root to Python path to enable imports without __init__.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Import our menu handlers
from core.utils.menu_handlers import (
	run_data_preparation,
	run_single_labeling_batch, 
	run_fine_tuning_menu,
	run_excel_export_menu,
	run_project_utilities_menu,
	print_manual_prompt
)
from core.labeling.label_manager import update_master_labels
from config.paths import ensure_directories

def main():
	"""Main function with interactive menu"""
	# Ensure all directories exist
	ensure_directories()
	
	consolidated_df = None  # Store DataFrame between operations
	
	while True:
		print("\n" + "=" * 60)
		print("TEXT CLASSIFICATION - MAIN MENU")
		print("=" * 60)
		print("1. Run data preparation")
		print("2. Run single labeling batch")
		print("3. Update master labels after corrections")
		print("4. Fine-tuning operations")
		print("5. Excel export & manual editing")
		print("6. Project utilities")
		print("0. Exit")
		print("=" * 60)

		choice = input("Enter your choice (0-6): ").strip()

		if choice == "1":
			try:
				consolidated_df = run_data_preparation()
				print(f"✅ Data preparation complete. DataFrame shape: {consolidated_df.shape}")
			except Exception as e:
				print(f"❌ Error in data preparation: {e}")

		elif choice == "2":
			try:
				if consolidated_df is None:
					print("❌ No DataFrame available. Run data preparation first.")
					continue
				
				print("\n" + "=" * 50)
				print("LABELING BATCH WORKFLOW")
				print("=" * 50)
				print("1. Generate TO_LABEL file with API predictions")
				print("2. Generate ChatGPT manual labeling prompt")
				print("0. Back to main menu")
				print("=" * 50)
				
				sub_choice = input("Enter your choice (0-2): ").strip()
				
				if sub_choice == "1":
					batch_size = int(input("Enter batch size (default 10): ") or "10")
					method = input("Texts selection method (longest/shortest/medium/random, default longest): ").strip() or "longest"
					max_examples = int(input("Max examples in prompt (default 30): ") or "30")
					run_single_labeling_batch(df=consolidated_df, batch_size=batch_size, selection_method=method, max_examples=max_examples)
				
				elif sub_choice == "2":
					print_manual_prompt()
				
				elif sub_choice == "0":
					continue
				
				else:
					print("Invalid choice")
					
			except Exception as e:
				print(f"❌ Error in labeling batch: {e}")
		
		elif choice == "3":
			try:
				# List available MANUAL_LABEL files
				from config.paths import PATHS
				outputs_dir = PATHS['manual_batches']
				to_label_files = [f for f in os.listdir(outputs_dir) if f.startswith('MANUAL_LABEL_') and '_AT-' in f] if os.path.exists(outputs_dir) else []
				if not to_label_files:
					print("No TO_LABEL files found")
					continue
				
				print("Available TO_LABEL files:")
				for i, file in enumerate(sorted(to_label_files), 1):
					print(f"  {i}. {file}")
				
				print(f"  {len(to_label_files) + 1}. Process ALL files automatically")
				
				file_choice = input(f"Select file (1-{len(to_label_files) + 1}): ").strip()
				
				if file_choice == str(len(to_label_files) + 1):
					# Process all files automatically - sort by timestamp from filename
					print(f"Processing all {len(to_label_files)} TO_LABEL files...")
					total_processed = 0
					
					# Sort files chronologically by timestamp in filename
					def extract_timestamp(filename):
						# Extract timestamp from MANUAL_LABEL filename pattern
						if '_AT-' in filename and 'MANUAL_LABEL_' in filename:
							# Pattern: MANUAL_LABEL_METHOD_AT-20250811_104333_NUM_EXAMPLES_IN_PROMPT_..
							try:
								timestamp_part = filename.split('_AT-')[1].split('_NUM_EXAMPLES_IN_PROMPT')[0]
								return timestamp_part
							except:
								return filename
						return filename
					
					sorted_files = sorted(to_label_files, key=extract_timestamp)
					
					for file in sorted_files:
						file_path = os.path.join(outputs_dir, file)
						print(f"Processing {file} (timestamp: {extract_timestamp(file)})...")
						try:
							master = update_master_labels(file_path)
							total_processed += 1
						except Exception as file_error:
							print(f"  ❌ Error processing {file}: {file_error}")
							import traceback
							traceback.print_exc()
					
					if total_processed > 0:
						print(f"✅ Processed {total_processed} files. Master labels now has {len(master)} total examples")
					else:
						print(f"❌ No files were processed successfully.")
				else:
					# Process single file
					file_choice = int(file_choice) - 1
					selected_file = os.path.join(outputs_dir, sorted(to_label_files)[file_choice])
					
					master = update_master_labels(selected_file)
					print(f"✅ Master labels updated with {len(master)} total examples")
				
			except Exception as e:
				print(f"❌ Error updating master labels: {e}")
		
		elif choice == "4":
			try:
				run_fine_tuning_menu(consolidated_df)
			except Exception as e:
				print(f"❌ Error in fine-tuning operations: {e}")
		
		elif choice == "5":
			try:
				run_excel_export_menu()
			except Exception as e:
				print(f"❌ Error in Excel export operations: {e}")
		
		elif choice == "6":
			try:
				run_project_utilities_menu()
			except Exception as e:
				print(f"❌ Error in project utilities: {e}")
		
		elif choice == "0":
			print("Goodbye!")
			break
		
		else:
			print("Invalid choice. Please try again.")


if __name__ == "__main__":
	main()