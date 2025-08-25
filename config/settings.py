"""
Configuration settings and constants - loads from environment variables
"""
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load configurations from environment variables
COLUMN_MAPPINGS = json.loads(os.getenv('COLUMN_MAPPINGS', '{}'))
FILES = json.loads(os.getenv('FILES', '{}'))
LABEL_INFO = json.loads(os.getenv('LABEL_INFO', '{}'))

ALL_LABELS = list(LABEL_INFO.keys())

# Default batch processing settings from environment
BATCH_DEFAULTS = {
    'size': int(os.getenv('BATCH_SIZE', '10')),
    'method': os.getenv('BATCH_METHOD', 'longest'),
    'max_examples': int(os.getenv('MAX_EXAMPLES', '30'))
}

# Load prompts from environment
PROMPTS = json.loads(os.getenv('PROMPTS', '{}'))
SYSTEM_PROMPT_TEMPLATE = os.getenv('SYSTEM_PROMPT_TEMPLATE', 'Generic text classifier')

def get_system_prompt():
    """Generate system prompt for classification"""
    if not LABEL_INFO:
        raise ValueError("LABEL_INFO not loaded from environment. Check .env file.")
    if not SYSTEM_PROMPT_TEMPLATE:
        raise ValueError("SYSTEM_PROMPT_TEMPLATE not loaded from environment. Check .env file.")
    definitions = "\n".join(f"• **{k}**: {v}" for k, v in LABEL_INFO.items())
    return SYSTEM_PROMPT_TEMPLATE.format(definitions=definitions)

def get_prompt(prompt_type, **kwargs):
    """Get a prompt template with formatting"""
    if not PROMPTS:
        raise ValueError("PROMPTS not loaded from environment. Check .env file.")
    if prompt_type not in PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(PROMPTS.keys())}")
    
    return PROMPTS[prompt_type].format(**kwargs)

SCHEMA = {
    "type": "object",
    "properties": {label: {"type": "integer", "enum": [0, 1]} for label in ALL_LABELS},
    "required": ALL_LABELS,
    "additionalProperties": False
}