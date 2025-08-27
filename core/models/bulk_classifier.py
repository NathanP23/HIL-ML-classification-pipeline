"""
Bulk classification functionality using fine-tuned models
"""
import json
import os
import pandas as pd
from openai import OpenAI
from config.settings import SCHEMA, ALL_LABELS, get_prompt
from config.paths import PATHS
from core.labeling.label_manager import load_labeled_ids, create_stable_ids


def classify_unlabeled_data_with_fine_tuned_model(client, fine_tuned_model, df, batch_size=50, include_manual=False):
	"""Classify data using fine-tuned model (unlabeled only or all data including manual)"""
	
	# Ensure DataFrame has stable IDs (same as in labeling.py)
	if 'id' not in df.columns:
		df = create_stable_ids(df.copy())
	
	# Get already labeled IDs for filtering or tracking
	labeled_ids = load_labeled_ids()
	
	if include_manual:
		# Classify all data (including manually labeled)
		target_df = df.copy()
		print(f"🔄 Including manually labeled data in classification")
		print(f"📊 Manual records to re-classify: {len(df[df['id'].isin(labeled_ids)])}")
	else:
		# Filter to unlabeled data only (original behavior)
		target_df = df[~df["id"].isin(labeled_ids)]
		if len(target_df) == 0:
			print("No unlabeled data found!")
			return []
	
	print(f"🤖 Fine-tuned model: {fine_tuned_model}")
	print(f"📊 Classifying {len(target_df)} total records...")
	print(f"📦 Processing in batches of {batch_size}")
	
	# Track which records were previously manual
	source_tracking = {}
	for _, row in target_df.iterrows():
		record_id = row['id']
		if record_id in labeled_ids:
			source_tracking[record_id] = 'previously_manual'
		else:
			source_tracking[record_id] = 'model'
	
	all_predictions = []
	
	# Process in batches
	for batch_start in range(0, len(target_df), batch_size):
		batch_end = min(batch_start + batch_size, len(target_df))
		batch_df = target_df.iloc[batch_start:batch_end]
		
		print(f"\n📦 Processing batch {batch_start//batch_size + 1}: "
			  f"Records {batch_start + 1}-{batch_end}")
		
		batch_predictions = []
		
		for i, (_, row) in enumerate(batch_df.iterrows(), 1):
			record_id = row['id']
			status_indicator = "🔄" if record_id in labeled_ids else "🆕"
			print(f"  {status_indicator} Classifying {batch_start + i}/{len(target_df)}: ID {record_id}")
			
			response = client.chat.completions.create(
				model=fine_tuned_model,
				messages=[
					{
						"role": "system",
						"content": get_prompt("system_basic")
					},
					{
						"role": "user",
						"content": get_prompt("user_with_json_instruction", text=row['text_content'])
					}
				],
				response_format={
					"type": "json_schema",
					"json_schema": {
						"strict": True,
						"name": "Classification",
						"schema": SCHEMA
					}
				}
			)
			
			prediction = json.loads(response.choices[0].message.content)
			
			# Combine row data with prediction
			result = {
				"id": row["id"],
				"text_content": row["text_content"]
			}
			result.update(prediction)
			batch_predictions.append(result)
		
		all_predictions.extend(batch_predictions)
		
		# Save intermediate results after each batch
		timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
		model_safe_name = fine_tuned_model.replace(':', '_').replace('/', '_')
		batch_dir = os.path.join(PATHS['base_output_dir'], "bulk_classification", "batches")
		os.makedirs(batch_dir, exist_ok=True)
		batch_file = os.path.join(batch_dir, f"BULK_CLASSIFICATION_BATCH_{batch_start//batch_size + 1}_{model_safe_name}_{timestamp}.json")
		
		with open(batch_file, 'w', encoding='utf-8') as f:
			json.dump(batch_predictions, f, ensure_ascii=False, indent=2)
		
		print(f"    ✅ Batch saved: {batch_file}")
	
	return all_predictions, source_tracking


def save_bulk_classification_results(predictions, fine_tuned_model, source_tracking=None):
	"""Save bulk classification results to final file"""
	timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
	model_safe_name = fine_tuned_model.replace(':', '_').replace('/', '_')
	
	# Add source info suffix if manual labels were included
	include_manual_suffix = ""
	if source_tracking:
		manual_count = sum(1 for source in source_tracking.values() if source == 'previously_manual')
		if manual_count > 0:
			include_manual_suffix = f"_WITH_MANUAL_{manual_count}"
	
	final_dir = os.path.join(PATHS['base_output_dir'], "bulk_classification", "final")
	os.makedirs(final_dir, exist_ok=True)
	final_file = os.path.join(final_dir, f"BULK_CLASSIFICATION_FINAL_{model_safe_name}_{timestamp}_TOTAL_{len(predictions)}{include_manual_suffix}.json")
	
	with open(final_file, 'w', encoding='utf-8') as f:
		json.dump(predictions, f, ensure_ascii=False, indent=2)
	
	print(f"\n🎉 BULK CLASSIFICATION COMPLETE!")
	print(f"   📁 Final results: {final_file}")
	print(f"   📊 Total classified: {len(predictions)} records")
	
	# Show source breakdown if manual labels were included
	if source_tracking:
		model_count = sum(1 for source in source_tracking.values() if source == 'model')
		manual_count = sum(1 for source in source_tracking.values() if source == 'previously_manual')
		print(f"   🆕 New model classifications: {model_count}")
		print(f"   🔄 Re-classified manual labels: {manual_count}")
	
	# Show category distribution
	print(f"\n📊 CLASSIFICATION SUMMARY:")
	category_counts = {}
	for label in ALL_LABELS:
		count = sum(1 for pred in predictions if pred.get(label, 0) == 1)
		category_counts[label] = count
		if count > 0:
			percentage = (count / len(predictions)) * 100
			print(f"   {label.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
	
	return final_file


def estimate_bulk_classification_cost(df, labeled_ids=None, include_manual=False):
	"""Estimate cost for bulk classification"""
	# Ensure DataFrame has stable IDs
	if 'id' not in df.columns:
		df = create_stable_ids(df.copy())
	
	if labeled_ids is None:
		labeled_ids = load_labeled_ids()
	
	if include_manual:
		# Count all records (including manual)
		total_count = len(df)
		manual_count = len(df[df["id"].isin(labeled_ids)])
		unlabeled_count = total_count - manual_count
		
		print(f"💰 BULK CLASSIFICATION COST ESTIMATE (INCLUDING MANUAL):")
		print(f"   📊 Total records: {total_count:,}")
		print(f"   📋 Manual records to re-classify: {manual_count:,}")
		print(f"   🆕 New records to classify: {unlabeled_count:,}")
		target_count = total_count
	else:
		# Count only unlabeled records (original behavior)
		unlabeled_count = len(df[~df["id"].isin(labeled_ids)])
		print(f"💰 BULK CLASSIFICATION COST ESTIMATE:")
		print(f"   📊 Unlabeled records: {unlabeled_count:,}")
		target_count = unlabeled_count
	
	# Rough cost estimation (check current OpenAI pricing)
	# Fine-tuned model costs are typically higher than base model
	cost_per_1k_tokens = 0.002  # Approximate - check actual pricing
	avg_tokens_per_record = 100  # Rough estimate based on text lengths
	
	estimated_tokens = target_count * avg_tokens_per_record
	estimated_cost = (estimated_tokens / 1000) * cost_per_1k_tokens
	
	print(f"   🔤 Estimated tokens: {estimated_tokens:,}")
	print(f"   💵 Estimated cost: ${estimated_cost:.2f}")
	print(f"   ⏱️  Estimated time: {target_count // 60:.0f}-{target_count // 30:.0f} minutes")
	
	return {
		"target_count": target_count,
		"unlabeled_count": unlabeled_count if not include_manual else unlabeled_count,
		"manual_count": len(df[df["id"].isin(labeled_ids)]) if include_manual else 0,
		"estimated_tokens": estimated_tokens,
		"estimated_cost": estimated_cost,
		"include_manual": include_manual
	}


def run_bulk_classification(client, fine_tuned_model, df, batch_size=50, include_manual=False):
	"""Complete bulk classification workflow"""
	
	# Estimate cost first
	title = "BULK CLASSIFICATION WITH FINE-TUNED MODEL"
	if include_manual:
		title += " (INCLUDING MANUAL LABELS)"
	print("=" * 80)
	print(title)
	print("=" * 80)
	
	cost_info = estimate_bulk_classification_cost(df, include_manual=include_manual)
	
	if cost_info["target_count"] == 0:
		print("No data to classify!")
		return None
	
	# Show what will be classified
	if include_manual:
		print(f"\n📋 CLASSIFICATION SCOPE:")
		print(f"   🆕 New records: {cost_info['unlabeled_count']:,}")
		print(f"   🔄 Manual records to re-classify: {cost_info['manual_count']:,}")
		print(f"   📊 Total to classify: {cost_info['target_count']:,}")
		confirm_msg = f"Proceed with classifying {cost_info['target_count']:,} records (including {cost_info['manual_count']:,} manual)? (y/N): "
	else:
		confirm_msg = f"Proceed with classifying {cost_info['target_count']:,} unlabeled records? (y/N): "
	
	# Confirm with user
	response = input(f"\n{confirm_msg}").strip().lower()
	if response != 'y':
		print("Classification cancelled.")
		return None
	
	# Create outputs directory
	os.makedirs(PATHS['base_output_dir'], exist_ok=True)
	
	# Run classification
	predictions, source_tracking = classify_unlabeled_data_with_fine_tuned_model(
		client, fine_tuned_model, df, batch_size, include_manual=include_manual
	)
	
	if not predictions:
		return None
	
	# Save results with source information
	final_file = save_bulk_classification_results(predictions, fine_tuned_model, source_tracking=source_tracking)
	
	return final_file, predictions, source_tracking