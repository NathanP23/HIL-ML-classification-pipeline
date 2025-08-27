"""
Fine-tuning functionality for OpenAI models
"""
import json
import os
import time
from openai import OpenAI
from config.paths import PATHS
from config.settings import get_prompt, ALL_LABELS


def create_fine_tune_job(client, training_file_id, model="gpt-4o-mini-2024-07-18", epochs=3, batch_size=1, learning_rate_multiplier=1.0, suffix=None):
	"""Create a fine-tuning job"""
	try:
		hyperparameters = {"n_epochs": epochs}
		if batch_size != 1:
			hyperparameters["batch_size"] = batch_size
		if learning_rate_multiplier != 1.0:
			hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier
			
		job_params = {
			"training_file": training_file_id,
			"model": model,
			"hyperparameters": hyperparameters
		}
		
		if suffix:
			job_params["suffix"] = suffix
			
		job = client.fine_tuning.jobs.create(**job_params)
		print(f"✅ Fine-tuning job created: {job.id}")
		print(f"📊 Model: {model}")
		print(f"📁 Training file: {training_file_id}")
		
		return job
	
	except Exception as e:
		print(f"❌ Error creating fine-tuning job: {e}")
		return None


def upload_training_file(client, jsonl_path=None):
	"""Upload JSONL file for fine-tuning"""
	if jsonl_path is None:
		jsonl_path = PATHS['ft_data']
	
	if not os.path.exists(jsonl_path):
		print(f"❌ JSONL file not found: {jsonl_path}")
		return None
	
	try:
		with open(jsonl_path, "rb") as f:
			response = client.files.create(
				file=f,
				purpose="fine-tune"
			)
		
		print(f"✅ Training file uploaded: {response.id}")
		print(f"📁 Local file: {jsonl_path}")
		
		return response.id
	
	except Exception as e:
		print(f"❌ Error uploading training file: {e}")
		return None


def monitor_fine_tuning_job(client, job_id):
	"""Monitor fine-tuning job progress"""
	try:
		while True:
			job = client.fine_tuning.jobs.retrieve(job_id)
			
			print(f"🔄 Status: {job.status}")
			
			if job.status == "succeeded":
				print(f"✅ Fine-tuning completed!")
				print(f"🤖 Model ID: {job.fine_tuned_model}")
				return job.fine_tuned_model
			
			elif job.status == "failed":
				print(f"❌ Fine-tuning failed")
				if job.error:
					print(f"Error: {job.error}")
				return None
			
			elif job.status in ["cancelled", "validating_files"]:
				print(f"⚠️ Job status: {job.status}")
				return None
			
			# Wait before checking again
			print("⏳ Waiting 30 seconds...")
			time.sleep(30)
	
	except Exception as e:
		print(f"❌ Error monitoring job: {e}")
		return None


def start_fine_tuning_workflow(client, model="gpt-4o-mini-2024-07-18"):
	"""Complete fine-tuning workflow"""
	print("🚀 STARTING FINE-TUNING WORKFLOW")
	print("=" * 50)
	
	# Step 1: Upload training file
	print("\n📤 Step 1: Uploading training file...")
	training_file_id = upload_training_file(client)
	
	if not training_file_id:
		return None
	
	# Step 2: Create fine-tuning job
	print("\n🎯 Step 2: Creating fine-tuning job...")
	job = create_fine_tune_job(client, training_file_id, model)
	
	if not job:
		return None
	
	# Step 3: Monitor job
	print("\n👀 Step 3: Monitoring job progress...")
	fine_tuned_model = monitor_fine_tuning_job(client, job.id)
	
	if fine_tuned_model:
		print(f"\n🎉 FINE-TUNING COMPLETED!")
		print(f"   🤖 Model ID: {fine_tuned_model}")
		print(f"   💡 You can now use this model for classification")
		
		return fine_tuned_model
	
	return None


def check_fine_tune_status(client, job_id):
	"""Check fine-tuning job status"""
	try:
		job = client.fine_tuning.jobs.retrieve(job_id)
		
		print(f"🔄 Job ID: {job_id}")
		print(f"📊 Status: {job.status}")
		print(f"📅 Created: {job.created_at}")
		
		if job.status == "succeeded":
			print(f"✅ Model ready: {job.fine_tuned_model}")
			return job.fine_tuned_model
		elif job.status == "failed":
			print(f"❌ Job failed")
			if job.error:
				print(f"Error: {job.error}")
		elif job.status in ["running", "validating_files"]:
			print(f"⏳ Job in progress...")
		
		return None
	
	except Exception as e:
		print(f"❌ Error checking status: {e}")
		return None


def list_fine_tune_jobs(client):
	"""List all fine-tuning jobs"""
	try:
		jobs = client.fine_tuning.jobs.list()
		
		print("🤖 FINE-TUNING JOBS:")
		print("-" * 60)
		
		for job in jobs.data:
			status_icon = "✅" if job.status == "succeeded" else "⏳" if job.status == "running" else "❌"
			print(f"{status_icon} {job.id}")
			print(f"   Status: {job.status}")
			print(f"   Model: {job.model}")
			if job.fine_tuned_model:
				print(f"   Fine-tuned: {job.fine_tuned_model}")
			print(f"   Created: {job.created_at}")
			print()
		
	except Exception as e:
		print(f"❌ Error listing jobs: {e}")


def estimate_fine_tuning_cost():
	"""Estimate fine-tuning cost based on training data"""
	try:
		jsonl_path = PATHS['ft_data']
		
		if not os.path.exists(jsonl_path):
			print(f"❌ Training file not found: {jsonl_path}")
			print("💡 Run master label updates first to generate training data")
			return
		
		# Count tokens in training data
		total_tokens = 0
		example_count = 0
		
		with open(jsonl_path, 'r', encoding='utf-8') as f:
			for line in f:
				if line.strip():
					data = json.loads(line)
					messages = data.get('messages', [])
					
					# Rough token estimation (1 token ≈ 4 characters)
					for message in messages:
						content = message.get('content', '')
						total_tokens += len(content) // 4
					
					example_count += 1
		
		# OpenAI pricing estimates (approximate)
		cost_per_1k_tokens = 0.0080  # $0.008 per 1K tokens for training
		estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
		
		print("💰 FINE-TUNING COST ESTIMATE:")
		print(f"   📊 Training examples: {example_count}")
		print(f"   🔤 Estimated tokens: {total_tokens:,}")
		print(f"   💵 Estimated cost: ${estimated_cost:.4f}")
		print(f"   💡 Actual cost may vary based on OpenAI pricing")
		
	except Exception as e:
		print(f"❌ Error estimating cost: {e}")


def test_fine_tuned_model_simple(model_name, client):
	"""Simple fine-tuned model test (wrapper)"""
	from core.models.evaluation import test_fine_tuned_model
	return test_fine_tuned_model(model_name, client)


def list_fine_tuned_models(client):
	"""List all fine-tuned models"""
	try:
		models = client.fine_tuning.jobs.list()
		
		print("🤖 FINE-TUNED MODELS:")
		print("-" * 40)
		
		for job in models.data:
			if job.fine_tuned_model:
				print(f"✅ {job.fine_tuned_model}")
				print(f"   Status: {job.status}")
				print(f"   Created: {job.created_at}")
				print()
		
		return [job.fine_tuned_model for job in models.data if job.fine_tuned_model]
	
	except Exception as e:
		print(f"❌ Error listing models: {e}")
		return []