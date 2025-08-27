"""
Few-shot learning functionality - system message creation with examples
"""
import os
import json
import glob
from config.settings import LABEL_INFO, ALL_LABELS, get_prompt
from config.paths import PATHS


def create_system_message_with_examples(labeled_json_file=None, max_examples=30):
	"""Create system message with progressive few-shot examples"""
	definitions = "\n".join(f"• {k}: {v}" for k, v in LABEL_INFO.items())
	
	# Load existing labeled examples as few-shot examples
	# Find the most recent TOTAL_MANUAL_LABEL file if no specific file provided
	if labeled_json_file is None:
		master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
		existing_master_files = glob.glob(master_pattern)
		if existing_master_files:
			labeled_json_file = sorted(existing_master_files)[-1]
	
	examples_text = ""
	if labeled_json_file and os.path.exists(labeled_json_file):
		with open(labeled_json_file, 'r', encoding='utf-8') as f:
			labeled_examples = json.load(f)
		
		# Limit examples to max_examples (take most recent ones)
		if len(labeled_examples) > max_examples:
			labeled_examples = labeled_examples[-max_examples:]
		
		if labeled_examples:
			examples_text = "\n\nExamples:\n"
			for i, example in enumerate(labeled_examples, 1):
				example_labels = []
				for label in ALL_LABELS:
					if example.get(label, 0) == 1:
						example_labels.append(label)
				
				labels_str = ", ".join(example_labels) if example_labels else "none"
				examples_text += f"{i}. Text: {example['text_content'][:100]}{'...' if len(example['text_content']) > 100 else ''}\n"
				examples_text += f"   Categories: {labels_str}\n\n"
	
	return get_prompt("system_classifier_with_examples", definitions=definitions, examples_text=examples_text)


def create_baseline_system_message():
	"""Create baseline system message without examples"""
	definitions = "\n".join(f"• {k}: {v}" for k, v in LABEL_INFO.items())
	
	return get_prompt("system_classifier_baseline", definitions=definitions)


def create_leave_one_out_system_message(excluded_id, max_examples=30):
	"""Create system message with examples excluding one specific ID"""
	definitions = "\n".join(f"• {k}: {v}" for k, v in LABEL_INFO.items())
	
	# Find the most recent master file
	master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
	existing_master_files = glob.glob(master_pattern)
	
	examples_text = ""
	if existing_master_files:
		latest_master_file = sorted(existing_master_files)[-1]
		with open(latest_master_file, 'r', encoding='utf-8') as f:
			labeled_examples = json.load(f)
		
		# Exclude the specified ID
		labeled_examples = [ex for ex in labeled_examples if ex['id'] != excluded_id]
		
		# Limit examples to max_examples
		if len(labeled_examples) > max_examples:
			labeled_examples = labeled_examples[-max_examples:]
		
		if labeled_examples:
			examples_text = "\n\nExamples:\n"
			for i, example in enumerate(labeled_examples, 1):
				example_labels = []
				for label in ALL_LABELS:
					if example.get(label, 0) == 1:
						example_labels.append(label)
				
				labels_str = ", ".join(example_labels) if example_labels else "none"
				examples_text += f"{i}. Text: {example['text_content'][:100]}{'...' if len(example['text_content']) > 100 else ''}\n"
				examples_text += f"   Categories: {labels_str}\n\n"
	
	return get_prompt("system_classifier_with_examples", definitions=definitions, examples_text=examples_text)


def load_existing_examples():
	"""Load existing examples from master labels"""
	master_pattern = os.path.join(PATHS['manual_master'], "TOTAL_MANUAL_LABEL_AT-*_TOTAL_SAMPLE_SIZE_*.json")
	existing_master_files = glob.glob(master_pattern)
	
	if existing_master_files:
		latest_master_file = sorted(existing_master_files)[-1]
		with open(latest_master_file, 'r', encoding='utf-8') as f:
			return json.load(f)
	
	return []


def format_examples_for_prompt(examples, max_examples=30):
	"""Format examples for prompt display"""
	if len(examples) > max_examples:
		examples = examples[-max_examples:]
	
	examples_text = ""
	for i, example in enumerate(examples, 1):
		example_labels = []
		for label in ALL_LABELS:
			if example.get(label, 0) == 1:
				example_labels.append(label)
		
		labels_str = ", ".join(example_labels) if example_labels else "none"
		examples_text += f"{i}. Text: {example['text_content'][:100]}{'...' if len(example['text_content']) > 100 else ''}\n"
		examples_text += f"   Categories: {labels_str}\n\n"
	
	return examples_text