"""
Batch processing functionality - batch selection and API classification
"""
import json
import os
import pandas as pd
from openai import OpenAI
from config.settings import SCHEMA, get_prompt
from config.paths import PATHS
from .prompt_builder import create_system_message_with_examples
from .label_manager import save_api_predictions


def select_batch_by_length(df, labeled_ids, batch_size=10):
    """Select batch of unlabeled records by longest text length"""
    unlabeled = df[~df["id"].isin(labeled_ids)]
    
    if len(unlabeled) == 0:
        print("No unlabeled records remaining!")
        return pd.DataFrame()
    
    text_column = 'text_content'
    # Select by longest text from unlabeled records only
    batch = unlabeled.loc[unlabeled[text_column].str.len().nlargest(min(batch_size, len(unlabeled))).index]
    
    print(f"Selected {len(batch)} records to label (longest texts first)")
    print(f"IDs to label: {batch['id'].tolist()}")
    
    return batch


def select_batch_by_shortest(df, labeled_ids, batch_size=10):
    """Select batch of unlabeled records by shortest text length"""
    unlabeled = df[~df["id"].isin(labeled_ids)]
    
    if len(unlabeled) == 0:
        print("No unlabeled records remaining!")
        return pd.DataFrame()
    
    text_column = 'text_content'
    # Select by shortest text from unlabeled records only
    batch = unlabeled.loc[unlabeled[text_column].str.len().nsmallest(min(batch_size, len(unlabeled))).index]
    
    print(f"Selected {len(batch)} records to label (shortest texts first)")
    print(f"IDs to label: {batch['id'].tolist()}")
    
    return batch


def select_batch_by_medium_length(df, labeled_ids, batch_size=10):
    """Select batch of unlabeled records closest to average text length"""
    unlabeled = df[~df["id"].isin(labeled_ids)]
    
    if len(unlabeled) == 0:
        print("No unlabeled records remaining!")
        return pd.DataFrame()
    
    text_column = 'text_content'
    # Calculate average length
    text_lengths = unlabeled[text_column].str.len()
    avg_length = text_lengths.mean()
    
    # Find records closest to average length
    unlabeled['distance_from_avg'] = abs(text_lengths - avg_length)
    batch = unlabeled.nsmallest(min(batch_size, len(unlabeled)), 'distance_from_avg')
    batch = batch.drop('distance_from_avg', axis=1)  # Remove helper column
    
    print(f"Selected {len(batch)} records to label (closest to average length: {avg_length:.1f} chars)")
    print(f"IDs to label: {batch['id'].tolist()}")
    
    return batch


def select_batch_random(df, labeled_ids, batch_size=10, random_state=42):
    """Select batch of unlabeled records randomly"""
    unlabeled = df[~df["id"].isin(labeled_ids)]
    
    if len(unlabeled) == 0:
        print("No unlabeled records remaining!")
        return pd.DataFrame()
    
    batch = unlabeled.sample(n=min(batch_size, len(unlabeled)), random_state=random_state)
    
    print(f"Selected {len(batch)} records to label (random selection)")
    print(f"IDs to label: {batch['id'].tolist()}")
    
    return batch


def classify_batch_with_api(batch, client, model="gpt-4.1-2025-04-14", max_examples=30):
    """Get API predictions for a batch of records"""
    if batch.empty:
        return []
    
    records = []
    system_message = create_system_message_with_examples(max_examples=max_examples)
    
    print(f"Using system message with {len(system_message)} characters")
    print(f"Number of existing examples in prompt: {system_message.count('Example ')}")
    print(f"Max examples allowed: {max_examples}")
    
    text_column = 'text_content'
    
    # Use the batch DataFrame
    total_records = len(batch)
    for i, (_, row) in enumerate(batch.iterrows(), 1):
        print(f"Classifying ID {row.id}... ({i}/{total_records})")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": get_prompt("user_basic", text=row[text_column])}
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
        rec = {"id": row.id, text_column: row[text_column]}
        rec.update(pred)
        records.append(rec)
    
    return records, resp.model


def process_batch_for_labeling(df, labeled_ids, client, batch_size=10, selection_method="longest", max_examples=30):
    """Complete batch processing pipeline"""
    # Select batch
    if selection_method == "longest" or selection_method == "length":  # Support both names
        batch = select_batch_by_length(df, labeled_ids, batch_size)
    elif selection_method == "shortest":
        batch = select_batch_by_shortest(df, labeled_ids, batch_size)
    elif selection_method == "medium":
        batch = select_batch_by_medium_length(df, labeled_ids, batch_size)
    else:
        batch = select_batch_random(df, labeled_ids, batch_size)
    
    if batch.empty:
        return None, None
    
    # Get API predictions
    records, model = classify_batch_with_api(batch, client, max_examples=max_examples)
    
    # Save predictions for manual review
    batches_dir = PATHS['manual_batches']
    os.makedirs(batches_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    method_name = selection_method.upper()
    to_label_file = os.path.join(batches_dir, f"MANUAL_LABEL_{method_name}_AT-{timestamp}_NUM_EXAMPLES_IN_PROMPT_{max_examples}_SAMPLE_SIZE_{batch_size}_MODEL-{model.replace(':', '_')}_.json")
    
    with open(to_label_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    # Save API predictions immediately (before manual correction)
    save_api_predictions(records)
    
    print(f"Predictions saved to `{to_label_file}`. Please correct values manually.")
    print("Next: After manual correction, use update_master_labels() to update the master examples file.")
    
    return to_label_file, records