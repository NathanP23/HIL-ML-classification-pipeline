"""
Model evaluation functionality - performance testing and metrics
"""
import json
import os
import glob
import pandas as pd
from openai import OpenAI
from config.settings import SCHEMA, ALL_LABELS, get_prompt
from config.paths import PATHS
from core.labeling.prompt_builder import create_system_message_with_examples, create_baseline_system_message, create_leave_one_out_system_message


def test_api_performance_baseline(labeled_json_file=None, client=None, model="gpt-4.1-2025-04-14"):
    """Test API performance without examples (baseline)"""
    if not client:
        raise ValueError("OpenAI client is required")
    
    # Find the most recent master labeled file if not specified
    if labeled_json_file is None:
        master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
        existing_master_files = glob.glob(master_pattern)
        if existing_master_files:
            labeled_json_file = sorted(existing_master_files)[-1]  # Use most recent
        else:
            raise FileNotFoundError("No labeled data found. Run manual labeling first.")
    
    # Load already labeled data
    with open(labeled_json_file, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    
    print(f"Testing API baseline on {len(labeled_data)} previously labeled records...")
    
    # Get API predictions without examples
    test_records = []
    system_message = create_baseline_system_message()
    
    print(f"Using baseline system message (no examples)")
    
    for i, record in enumerate(labeled_data):
        print(f"Baseline classifying {i+1}/{len(labeled_data)}: ID {record['id']}...")
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": get_prompt("user_basic", text=record['text_content'])}
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
        
        pred = json.loads(resp.choices[0].message.content)
        test_rec = {"id": record['id'], "text_content": record['text_content']}
        test_rec.update(pred)
        test_records.append(test_rec)
    
    # Save baseline predictions
    eval_dir = os.path.join(PATHS['base_output_dir'], "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "test_baseline.json"), 'w', encoding='utf-8') as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
    
    # Calculate accuracy metrics
    category_stats, overall_accuracy = calculate_accuracy_metrics(labeled_data, test_records, "BASELINE")
    
    return category_stats, overall_accuracy


def test_api_performance_leave_one_out(labeled_json_file=None, client=None, model="gpt-4.1-2025-04-14"):
    """Test API performance using leave-one-out validation"""
    if not client:
        raise ValueError("OpenAI client is required")
    
    # Find the most recent master labeled file if not specified
    if labeled_json_file is None:
        master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
        existing_master_files = glob.glob(master_pattern)
        if existing_master_files:
            labeled_json_file = sorted(existing_master_files)[-1]  # Use most recent
        else:
            raise FileNotFoundError("No labeled data found. Run manual labeling first.")
    
    # Load already labeled data
    with open(labeled_json_file, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    
    print(f"Testing API with leave-one-out on {len(labeled_data)} records...")
    
    test_records = []
    
    for i, record in enumerate(labeled_data):
        print(f"Leave-one-out classifying {i+1}/{len(labeled_data)}: ID {record['id']}...")
        
        # Create system message excluding current record by ID
        system_message = create_leave_one_out_system_message(record['id'])
        
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": get_prompt("user_basic", text=record['text_content'])}
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
        
        pred = json.loads(resp.choices[0].message.content)
        test_rec = {"id": record['id'], "text_content": record['text_content']}
        test_rec.update(pred)
        test_records.append(test_rec)
    
    # Save leave-one-out predictions
    eval_dir = os.path.join(PATHS['base_output_dir'], "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "test_leave_one_out.json"), 'w', encoding='utf-8') as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
    
    # Calculate accuracy metrics
    category_stats, overall_accuracy = calculate_accuracy_metrics(labeled_data, test_records, "LEAVE-ONE-OUT")
    
    return category_stats, overall_accuracy


def calculate_accuracy_metrics(labeled_data, test_records, test_name="TEST"):
    """Calculate and display accuracy metrics"""
    print("\n" + "="*60)
    print(f"{test_name} ACCURACY METRICS")
    print("="*60)
    
    total_correct = 0
    total_predictions = 0
    category_stats = {}
    
    for category in ALL_LABELS:
        correct = 0
        total = 0
        
        for i, labeled_record in enumerate(labeled_data):
            test_record = test_records[i]
            
            # Compare predictions
            labeled_value = labeled_record.get(category, 0)
            api_value = test_record.get(category, 0)
            
            if labeled_value == api_value:
                correct += 1
                total_correct += 1
            
            total += 1
            total_predictions += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        category_stats[category] = {"correct": correct, "total": total, "accuracy": accuracy}
        
        print(f"{category:35} {correct:3}/{total:3} ({accuracy:5.1f}%)")
    
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n{'OVERALL ACCURACY':35} {total_correct:3}/{total_predictions:3} ({overall_accuracy:5.1f}%)")
    
    return category_stats, overall_accuracy


def test_fine_tuned_model(fine_tuned_model, client=None, master_file=None):
    """Test fine-tuned model against your manually labeled examples"""
    if not client:
        raise ValueError("OpenAI client is required")
    
    # Find most recent master file if not specified
    if master_file is None:
        import glob
        master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
        master_files = glob.glob(master_pattern)
        if not master_files:
            raise FileNotFoundError("No master labeled files found. Run manual labeling first.")
        master_file = sorted(master_files)[-1]
    
    # Load manually labeled examples
    with open(master_file, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    
    print(f"Testing fine-tuned model: {fine_tuned_model}")
    print(f"Using labeled data: {master_file}")
    print(f"Testing on {len(labeled_data)} manually labeled records...")
    
    # Get fine-tuned model predictions
    test_records = []
    
    for i, record in enumerate(labeled_data):
        print(f"Fine-tuned model classifying {i+1}/{len(labeled_data)}: ID {record['id']}...")
        
        resp = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[
                {
                    "role": "system",
                    "content": get_prompt("system_basic")
                },
                {
                    "role": "user",
                    "content": get_prompt("user_with_json_instruction", text=record['text_content'])
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
        
        pred = json.loads(resp.choices[0].message.content)
        test_rec = {"id": record['id'], "text_content": record['text_content']}
        test_rec.update(pred)
        test_records.append(test_rec)
    
    # Save fine-tuned model predictions
    model_safe_name = fine_tuned_model.replace(':', '_').replace('/', '_')
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(PATHS['base_output_dir'], "fine_tuning", "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, f"test_fine_tuned_{model_safe_name}_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
    
    print(f"Fine-tuned model predictions saved to: {output_file}")
    
    # Calculate detailed accuracy metrics
    category_stats, overall_accuracy = calculate_detailed_accuracy_metrics(labeled_data, test_records, f"FINE-TUNED MODEL")
    
    return category_stats, overall_accuracy


def calculate_detailed_accuracy_metrics(labeled_data, test_records, test_name="TEST"):
    """Calculate detailed accuracy metrics with per-category analysis"""
    print("\n" + "="*80)
    print(f"{test_name} - DETAILED ACCURACY METRICS")
    print("="*80)
    
    total_correct = 0
    total_predictions = 0
    category_stats = {}
    
    # Header for detailed output
    print(f"{'Category':<35} {'Correct':<8} {'Total':<8} {'Accuracy':<10} {'TP':<4} {'FP':<4} {'FN':<4} {'TN':<4}")
    print("-"*80)
    
    for category in ALL_LABELS:
        correct = 0
        total = 0
        tp = fp = fn = tn = 0  # True/False Positive/Negative
        
        for i, labeled_record in enumerate(labeled_data):
            test_record = test_records[i]
            
            # Get actual and predicted values
            actual = labeled_record.get(category, 0)
            predicted = test_record.get(category, 0)
            
            # Calculate confusion matrix elements
            if actual == 1 and predicted == 1:
                tp += 1
                correct += 1
            elif actual == 0 and predicted == 0:
                tn += 1
                correct += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1
            
            total += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        category_stats[category] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": tp + fn  # Total positive examples
        }
        
        total_correct += correct
        total_predictions += total
        
        # Display row
        category_display = category.replace('_', ' ').title()[:34]
        print(f"{category_display:<35} {correct:<8} {total:<8} {accuracy:<10.1f} {tp:<4} {fp:<4} {fn:<4} {tn:<4}")
    
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
    
    print("-"*80)
    print(f"{'OVERALL ACCURACY':<35} {total_correct:<8} {total_predictions:<8} {overall_accuracy:<10.1f}")
    
    # Show categories with poor performance
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    poor_categories = [cat for cat, stats in category_stats.items() if stats['f1_score'] < 0.7]
    good_categories = [cat for cat, stats in category_stats.items() if stats['f1_score'] >= 0.8]
    
    print(f"   ✅ Excellent (F1 >= 0.8): {len(good_categories)} categories")
    print(f"   ⚠️  Needs improvement (F1 < 0.7): {len(poor_categories)} categories")
    
    if poor_categories:
        print(f"\n   Categories needing attention:")
        for cat in poor_categories[:5]:  # Show top 5 problematic
            stats = category_stats[cat]
            print(f"   - {cat.replace('_', ' ').title()}: F1={stats['f1_score']:.3f}, "
                  f"Precision={stats['precision']:.3f}, Recall={stats['recall']:.3f}")
    
    return category_stats, overall_accuracy


def compare_api_vs_manual_corrections(api_predictions_file=None, labeled_json_file=None):
    """Compare original API predictions vs manual corrections"""
    
    # Use centralized paths if not specified
    if api_predictions_file is None:
        api_predictions_file = PATHS['api_predictions']
    
    if labeled_json_file is None:
        # Find the most recent master labeled file
        master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
        existing_master_files = glob.glob(master_pattern)
        if existing_master_files:
            labeled_json_file = sorted(existing_master_files)[-1]  # Use most recent
        else:
            raise FileNotFoundError("No labeled data found. Run manual labeling first.")
    
    # Load both files
    with open(api_predictions_file, 'r', encoding='utf-8') as f:
        api_data = json.load(f)
    
    with open(labeled_json_file, 'r', encoding='utf-8') as f:
        manual_data = json.load(f)
    
    # Create lookup dictionary for manual corrections
    manual_lookup = {item['id']: item for item in manual_data}
    
    # Find matching records
    matching_records = []
    for api_record in api_data:
        if api_record['id'] in manual_lookup:
            matching_records.append((api_record, manual_lookup[api_record['id']]))
    
    print(f"Found {len(matching_records)} matching records between API predictions and manual corrections")
    
    if not matching_records:
        print("No matching records found for comparison")
        return
    
    # Calculate accuracy metrics
    category_stats, overall_accuracy = calculate_accuracy_metrics(
        [manual for api, manual in matching_records],
        [api for api, manual in matching_records],
        "API vs MANUAL"
    )
    
    return category_stats, overall_accuracy