# HIL ML Classification Pipeline

A human-in-the-loop machine learning system for automated text classification using progressive few-shot learning. Features interactive workflows for data preparation, model training with OpenAI API, human feedback integration, and bulk classification.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HIL-ML-classification-pipeline
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your OpenAI API key and configuration
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

- **OPENAI_API_KEY**: Your OpenAI API key
- **COLUMN_MAPPINGS**: JSON mapping of standardized to original column names
- **FILES**: JSON configuration for input data files
- **LABEL_INFO**: JSON definitions of classification categories
- **PROMPTS**: JSON templates for different prompts
- **SYSTEM_PROMPT_TEMPLATE**: Template for system prompts

## 📋 Overview

This project provides an interactive text classification system using GPT-4 with progressive few-shot learning. The system processes text documents and classifies them into predefined categories using human-in-the-loop feedback to continuously improve accuracy.

## 🎛️ Menu Options

The system provides an interactive menu with the following options:

### Option 1: Run data preparation

**Purpose:** Load, anonymize, and consolidate raw data into a single DataFrame ready for classification.

**Options Available:**
1. **Load from raw Excel files** - Processes original data with full anonymization pipeline
2. **Load from anonymized CSV files** - Uses previously processed and anonymized CSV files

**Functions Called:**
```
core.utils.menu_handlers.run_data_preparation()
├── **Purpose:** Orchestrates the complete data preparation workflow with user choice
├── **Args:** None
├── **Returns:** pd.DataFrame - Consolidated DataFrame ready for classification
│
├── **Option 1 Path:** Load from raw Excel files
│   ├── core.data.loader.load_data()
│   │   ├── **Purpose:** Load Excel files and apply column mappings
│   │   ├── **Args:** None (reads from config.settings.FILES and COLUMN_MAPPINGS)
│   │   ├── **Returns:** dict - {"df1": DataFrame, "df2": DataFrame, "df3": DataFrame}
│   │   └── **Process:** Reads Excel files → Applies column name standardization → Returns mapped DataFrames
│   │
│   └── core.data.anonymizer.anonymize_data(dfs)
│       ├── **Purpose:** Apply anonymization to sensitive data fields
│       ├── **Args:** dfs: dict - Dictionary of DataFrames from load_data()
│       ├── **Returns:** dict - Same structure with anonymized data
│       └── **Process:**
│           ├── anonymize_national_id(text) - Replace ID numbers with sequential digits
│           ├── anonymize_phone_numbers(text) - Standardize phone number format
│           └── anonymize_names_from_excel(text) - Replace names with placeholders
│
├── **Option 2 Path:** Load from anonymized CSV files
│   └── core.utils.menu_handlers.load_from_anonymized_csv()
│       ├── **Purpose:** Load previously processed anonymized CSV files
│       ├── **Args:** None (reads from anonymized/ directory)
│       ├── **Returns:** dict - {"df1": DataFrame, "df2": DataFrame, "df3": DataFrame}
│       └── **Process:** Loads df1_anonymized.csv, df2_anonymized.csv, df3_anonymized.csv
│
└── core.data.loader.consolidate_data(all_df)
    ├── **Purpose:** Merge multiple DataFrames into single consolidated dataset
    ├── **Args:** all_df: dict - Anonymized DataFrames
    ├── **Returns:** pd.DataFrame - Consolidated DataFrame with deduplication
    └── **Process:** Extract text_content → Remove invalid entries → Group duplicates → Count appearances
```

**Output Files:**
- **Location:** `anonymized/`
- **Format:** CSV files
- **Files:** `df1_anonymized.csv`, `df2_anonymized.csv`, `df3_anonymized.csv`
- **Content:** Original data with sensitive information anonymized
- **In-Memory:** Consolidated DataFrame with unique text records ready for labeling

---

### Option 2: Run single labeling batch

**Purpose:** Select unlabeled texts, get API predictions, and prepare them for manual correction.

**Execution Flow:**
```
main.py: run_single_labeling_batch()
├── Check if consolidated_df exists in memory
│
├── core.labeling.label_manager: prepare_for_labeling()
│   ├── create_stable_ids(): Create MD5 hash IDs from text_content
│   ├── load_labeled_ids(): Extract set of already labeled IDs
│   └── Return DataFrame with hash IDs + labeled_ids set
│
├── core.labeling.batch_processor: process_batch_for_labeling()
│   ├── Select batch method (longest/shortest/medium/random)
│   │   ├── select_batch_by_length() - Select by longest text length
│   │   ├── select_batch_by_shortest() - Select by shortest text length  
│   │   ├── select_batch_by_medium_length() - Select by closest to average length
│   │   ├── select_batch_random() - Random selection with seed
│   │   ├── Filter: unlabeled = df[~df["id"].isin(labeled_ids)]
│   │   └── Return batch DataFrame with selected records
│   │
│   ├── classify_batch_with_api()
│   │   ├── core.labeling.prompt_builder: create_system_message_with_examples()
│   │   │   ├── Load existing examples from master labels
│   │   │   ├── Create definitions from config.settings: LABEL_INFO
│   │   │   ├── Add recent examples to prompt (configurable max)
│   │   │   └── Return system message with few-shot examples
│   │   │
│   │   ├── For each text in batch:
│   │   │   ├── Call OpenAI API with system message + user text
│   │   │   ├── Use JSON schema from config.settings: SCHEMA
│   │   │   ├── Parse response to get category predictions (0/1)
│   │   │   └── Append to records list
│   │   └── Return predictions + model name
│   │
│   └── Save batch file for manual correction
```

**Output:** 
- New MANUAL_LABEL_METHOD_AT-*.json file for manual correction
- Updated outputs/api_predictions.json with raw API predictions
- Instructions to manually correct the TO_LABEL file

---

### Option 3: Update master labels after corrections

**Purpose:** Process manually corrected labels and add them to the training dataset.

**Functions Called:**
```
main.py: Menu Option 3 Handler
├── List available MANUAL_LABEL files in outputs/manual_labeling/batches/
├── User selection (single file or ALL files)
│
└── core.labeling.label_manager.update_master_labels(file_path)
    ├── **Purpose:** Integrate corrected batch into master training dataset
    ├── **Args:** file_path: str - Path to corrected MANUAL_LABEL file
    ├── **Returns:** list - Updated master labels dataset
    ├── **Process:**
    │   ├── Load corrected batch file (JSON format)
    │   ├── Load existing master labels from outputs/manual_labeling/master/TOTAL_MANUAL_LABEL_*.json
    │   ├── Merge new corrections with existing (deduplicate by ID)
    │   ├── Save consolidated master labels
    │   │
    │   └── Generate fine-tuning dataset:
    │       ├── core.labeling.prompt_builder.create_system_message_with_examples()
    │       ├── For each labeled example:
    │       │   ├── Create OpenAI training format (system, user, assistant)
    │       │   └── Append to JSONL training data
    │       └── Save outputs/fine_tuning/ft_data.jsonl
    │
    └── **Batch Processing Option:**
        ├── Process ALL files automatically
        ├── Sort files chronologically by timestamp
        └── Sequential processing maintaining data integrity
```

**Output Files:**
- **Location:** `outputs/manual_labeling/master/`
- **Format:** JSON and JSONL
- **Files:**
  - `TOTAL_MANUAL_LABEL_AT-TIMESTAMP_TOTAL_SAMPLE_SIZE_N.json` - Master training dataset
  - `outputs/fine_tuning/training/ft_data.jsonl` - OpenAI fine-tuning format
- **Content:** Human-corrected labels ready for model training

---

### Option 4: Fine-tuning operations

**Purpose:** Comprehensive model fine-tuning, evaluation, and bulk classification operations.

**Functions Called:**
```
core.utils.menu_handlers.run_fine_tuning_menu(consolidated_df)
├── **Sub-Option 1:** Estimate fine-tuning cost
│   └── core.models.fine_tuning.estimate_fine_tuning_cost()
│       ├── **Purpose:** Calculate OpenAI fine-tuning costs
│       ├── **Args:** None (reads from ft_data.jsonl)
│       └── **Returns:** Cost estimation display
│
├── **Sub-Option 2:** Upload training data and start fine-tuning
│   ├── core.models.fine_tuning.upload_training_file(client)
│   │   ├── **Purpose:** Upload JSONL to OpenAI
│   │   ├── **Args:** client: OpenAI - API client
│   │   └── **Returns:** str - File ID for training
│   │
│   └── core.models.fine_tuning.create_fine_tune_job(client, file_id, **params)
│       ├── **Purpose:** Start OpenAI fine-tuning job
│       ├── **Args:** client, file_id, epochs, batch_size, learning_rate_multiplier, suffix
│       └── **Returns:** Job object with job.id
│
├── **Sub-Option 3:** Check fine-tuning job status
│   └── core.models.fine_tuning.check_fine_tune_status(client, job_id)
│       ├── **Purpose:** Monitor training progress
│       ├── **Args:** client: OpenAI, job_id: str
│       └── **Returns:** str - Model name if completed
│
├── **Sub-Option 4:** List all fine-tuning jobs
│   └── core.models.fine_tuning.list_fine_tune_jobs(client)
│       ├── **Purpose:** Display all training jobs and status
│       ├── **Args:** client: OpenAI
│       └── **Returns:** Formatted job listing
│
├── **Sub-Option 5:** Test fine-tuned model
│   └── core.models.evaluation.test_fine_tuned_model(model_name, client)
│       ├── **Purpose:** Evaluate model against manual labels
│       ├── **Args:** model_name: str, client: OpenAI
│       └── **Returns:** tuple(stats: dict, accuracy: float)
│
├── **Sub-Option 6:** Bulk classify unlabeled data
│   └── core.models.bulk_classifier.run_bulk_classification(client, model_name, df, batch_size, include_manual)
│       ├── **Purpose:** Apply model to large datasets
│       ├── **Args:** client, model_name: str, df: DataFrame, batch_size: int, include_manual: bool
│       ├── **Returns:** tuple(file_path: str, predictions: list, source_tracking: dict)
│       └── **Process:**
│           ├── Filter unlabeled records (or all if include_manual=True)
│           ├── Batch processing with progress tracking
│           ├── Apply fine-tuned model predictions
│           └── Save timestamped results file
│
└── **Sub-Option 7:** Model performance comparison
    ├── core.models.evaluation.test_api_performance_baseline(client)
    ├── core.models.evaluation.test_api_performance_leave_one_out(client)
    └── core.models.evaluation.compare_api_vs_manual_corrections()
```

**Output Files:**
- **Location:** `outputs/fine_tuning/`
- **Format:** JSONL for training, JSON for results
- **Files:**
  - `ft_data.jsonl` - OpenAI training format
  - `bulk_classification/BULK_model-name_YYYY-MM-DD_HHMMSS.json` - Classification results
- **Content:** Model training data and bulk classification results

---

### Option 5: Excel export & manual editing

**Purpose:** Export classifications to Excel for manual review and import changes back to system.

**Functions Called:**
```
core.utils.menu_handlers.run_excel_export_menu()
├── **Sub-Option 1:** Export latest bulk results to Excel
│   └── core.utils.excel_export.export_latest_bulk_results(include_manual_labels=False)
│       ├── **Purpose:** Convert bulk classification JSON to Excel
│       ├── **Args:** include_manual_labels: bool
│       ├── **Returns:** str - Excel file path
│       └── **Features:** RTL layout, freeze panes, formatted columns
│
├── **Sub-Option 2:** Export bulk + manual labels combined
│   └── core.utils.excel_export.export_latest_bulk_results(include_manual_labels=True)
│       ├── **Purpose:** Merge bulk predictions with manual corrections
│       ├── **Args:** include_manual_labels=True
│       ├── **Returns:** str - Excel file path
│       └── **Features:** Source column shows manual vs model classifications
│
├── **Sub-Option 3:** Export manual labels only
│   └── core.utils.excel_export.export_manual_labels()
│       ├── **Purpose:** Export only human-corrected labels
│       ├── **Args:** None
│       ├── **Returns:** str - Excel file path
│       └── **Content:** Pure manual corrections from master labels
│
├── **Sub-Option 4:** Export custom JSON file
│   └── core.utils.excel_export.convert_json_to_excel_rtl(json_path, output_path, include_manual)
│       ├── **Purpose:** Convert any JSON classification file to Excel
│       ├── **Args:** json_path: str, output_path: str, include_manual: bool
│       ├── **Returns:** str - Excel file path
│       └── **Features:** Flexible JSON to Excel conversion
│
└── **Sub-Option 5:** Process Excel changes back to JSON
    └── core.utils.change_detection.process_excel_changes(original_json, excel_path, integrate)
        ├── **Purpose:** Import Excel edits back to classification system
        ├── **Args:** original_json: str, excel_path: str, integrate: bool
        ├── **Returns:** tuple(changes_json: str, master_file: str)
        └── **Process:**
            ├── Compare original JSON vs modified Excel
            ├── Detect classification changes
            ├── Generate change report
            └── Optionally integrate with master labels system
```

**Output Files:**
- **Location:** `outputs/excel_exports/`
- **Format:** Excel (.xlsx) and JSON
- **Files:**
  - `bulk_results_YYYY-MM-DD_HHMMSS.xlsx` - Bulk classifications
  - `manual_labels_YYYY-MM-DD_HHMMSS.xlsx` - Manual corrections
  - `changes_YYYY-MM-DD_HHMMSS.json` - Change detection results
- **Features:** RTL support, color coding, source tracking, freeze panes

---

### Option 6: Project utilities

**Purpose:** Project status monitoring, file maintenance, and system diagnostics.

**Functions Called:**
```
core.utils.menu_handlers.run_project_utilities_menu()
├── **Sub-Option 1:** Print project status
│   └── core.utils.project_status.print_project_status()
│       ├── **Purpose:** Display comprehensive project statistics
│       ├── **Args:** None
│       ├── **Returns:** Console output with statistics
│       └── **Displays:**
│           ├── Data files count and sizes
│           ├── Manual labels progress
│           ├── Batch processing history
│           ├── Fine-tuning dataset statistics
│           └── Recent file activity
│
├── **Sub-Option 2:** Cleanup old files
│   └── core.utils.file_ops.cleanup_old_files(keep)
│       ├── **Purpose:** Remove old batch files to free disk space
│       ├── **Args:** keep: int - Number of recent files to preserve
│       ├── **Returns:** Cleanup summary
│       └── **Process:**
│           ├── Identify old TO_LABEL and batch files
│           ├── Preserve specified number of recent files
│           ├── Remove outdated temporary files
│           └── Report space freed
│
└── **Sub-Option 3:** Export progress report
    └── core.utils.project_status.export_progress_report()
        ├── **Purpose:** Generate detailed project progress report
        ├── **Args:** None
        ├── **Returns:** str - Report file path
        └── **Content:**
            ├── Classification accuracy trends
            ├── Data processing timeline
            ├── Model performance metrics
            └── File system status
```

**Output Files:**
- **Location:** `outputs/reports/`
- **Format:** JSON and text reports
- **Files:**
  - `progress_report_YYYY-MM-DD_HHMMSS.json` - Detailed metrics
  - `project_status.txt` - Human-readable summary
- **Content:** Project statistics, file counts, accuracy trends, system health

---

### Option 0: Exit

**Purpose:** Gracefully exit the application, ensuring all data is properly saved and connections are closed.

---

## 📁 Project Structure

```
project/
├── main.py                    # Interactive menu system
│   ├── main() -> None                           # Main menu loop
│   ├── print_manual_prompt() -> None            # Manual labeling prompt display
│   └── ensure_directories() -> None             # Directory structure validation
│
├── requirements.txt           # Python dependencies (55 packages)
├── .env.example              # Environment configuration template
├── .gitignore                # Git ignore rules (exclude all, allow .py)
├── README.md                 # Project documentation
│
├── config/                   # Configuration modules
│   ├── settings.py          # Settings loaded from environment
│   │   ├── get_system_prompt() -> str            # Dynamic system prompt builder
│   │   ├── get_prompt(prompt_type, **kwargs) -> str # Get formatted prompt template
│   │   └── Environment Variables: OPENAI_API_KEY, COLUMN_MAPPINGS, FILES, LABEL_INFO, PROMPTS
│   │
│   └── paths.py             # File path configurations
│       ├── ensure_directories() -> None          # Create required directories
│       └── PATHS: dict                          # Central path configuration
│
├── core/                    # Core application modules
│   │
│   ├── data/                # Data processing modules
│   │   ├── anonymizer.py    # Data anonymization functions
│   │   │   ├── anonymize_data(dfs: dict) -> dict            # Main anonymization orchestrator
│   │   │   ├── anonymize_national_id(text: str) -> str      # Replace ID numbers with sequential digits
│   │   │   ├── anonymize_phone_numbers(text: str) -> str    # Standardize phone format
│   │   │   └── anonymize_names_from_excel(text: str) -> str # Replace names with placeholders
│   │   │
│   │   └── loader.py        # Data loading and consolidation
│   │       ├── load_data() -> dict                          # Load Excel files with column mapping
│   │       ├── consolidate_data(all_df: dict) -> DataFrame  # Merge DataFrames with deduplication
│   │       └── (No additional validation functions)
│   │
│   ├── labeling/            # Labeling workflow modules
│   │   ├── batch_processor.py    # Batch selection and API calls
│   │   │   ├── process_batch_for_labeling(df, labeled_ids, client, batch_size, selection_method, max_examples) -> tuple
│   │   │   ├── select_batch_by_length(df, labeled_ids, batch_size=10) -> DataFrame
│   │   │   ├── select_batch_by_shortest(df, labeled_ids, batch_size=10) -> DataFrame
│   │   │   ├── select_batch_by_medium_length(df, labeled_ids, batch_size=10) -> DataFrame
│   │   │   ├── select_batch_random(df, labeled_ids, batch_size=10, random_state=42) -> DataFrame
│   │   │   └── classify_batch_with_api(batch, client, model, max_examples) -> tuple
│   │   │
│   │   ├── label_manager.py      # Label management and ID creation
│   │   │   ├── prepare_for_labeling(df: DataFrame) -> tuple  # Add stable IDs, load existing labels
│   │   │   ├── create_stable_ids(df: DataFrame) -> DataFrame # Generate MD5 hash IDs from content
│   │   │   ├── load_labeled_ids() -> set                    # Extract already labeled ID set
│   │   │   └── update_master_labels(file_path: str) -> list # Integrate corrections into training dataset
│   │   │
│   │   └── prompt_builder.py     # Few-shot prompt creation
│   │       ├── create_system_message_with_examples(max_examples: int) -> str # Build few-shot system prompt
│   │       ├── load_existing_examples() -> list             # Load training examples from master labels
│   │       └── format_examples_for_prompt(examples: list) -> str # Format examples for API prompt
│   │
│   ├── models/              # Model and evaluation modules
│   │   ├── bulk_classifier.py    # Bulk classification functionality
│   │   │   ├── run_bulk_classification(client, model_name, df, batch_size, include_manual) -> tuple
│   │   │   ├── filter_unlabeled_data(df, include_manual) -> DataFrame
│   │   │   └── batch_classify_with_progress(client, model_name, df, batch_size) -> list
│   │   │
│   │   ├── evaluation.py         # Performance testing
│   │   │   ├── test_api_performance_baseline(client) -> tuple        # No examples evaluation
│   │   │   ├── test_api_performance_leave_one_out(client) -> tuple   # Fair few-shot testing
│   │   │   ├── compare_api_vs_manual_corrections() -> tuple          # Original vs corrected comparison
│   │   │   └── test_fine_tuned_model(model_name, client) -> tuple    # Fine-tuned model evaluation
│   │   │
│   │   └── fine_tuning.py        # Model fine-tuning
│   │       ├── upload_training_file(client) -> str                   # Upload JSONL to OpenAI
│   │       ├── create_fine_tune_job(client, file_id, **params) -> Job # Start fine-tuning job
│   │       ├── check_fine_tune_status(client, job_id) -> str         # Monitor training progress
│   │       ├── list_fine_tune_jobs(client) -> None                   # Display all jobs
│   │       └── estimate_fine_tuning_cost() -> None                   # Calculate training costs
│   │
│   └── utils/               # Utility modules
│       ├── change_detection.py   # File change monitoring
│       │   └── process_excel_changes(original_json, excel_path, integrate) -> tuple
│       │
│       ├── excel_export.py       # Excel export functionality
│       │   ├── export_latest_bulk_results(include_manual_labels) -> str
│       │   ├── export_manual_labels() -> str
│       │   └── convert_json_to_excel_rtl(json_path, output_path, include_manual) -> str
│       │
│       ├── file_ops.py           # File operations
│       │   └── cleanup_old_files(keep: int) -> None
│       │
│       ├── menu_handlers.py      # Menu system handlers
│       │   ├── run_data_preparation() -> DataFrame
│       │   ├── run_single_labeling_batch(df, batch_size, selection_method, max_examples) -> str
│       │   ├── run_fine_tuning_menu(consolidated_df) -> None
│       │   ├── run_excel_export_menu() -> None
│       │   └── run_project_utilities_menu() -> None
│       │
│       ├── project_status.py     # Project status reporting
│       │   ├── print_project_status() -> None
│       │   └── export_progress_report() -> str
│       │
│       └── tree_logger.py        # Logging utilities
│           ├── log(msg: str) -> None
│           └── print_tree(node: str, prefix: str) -> None
│
└── outputs/                 # Generated files (git-ignored)
    ├── bulk_classification/ # Bulk processing results
    │   └── BULK_model-name_YYYY-MM-DD_HHMMSS.json
    │
    ├── fine_tuning/         # Fine-tuning datasets
    │   └── ft_data.jsonl    # OpenAI training format
    │
    ├── manual_labeling/     # Manual labeling batches
    │   ├── batches/         # Individual batch files
    │   │   └── MANUAL_LABEL_METHOD_AT-YYYYMMDD_HHMMSS_*.json
    │   └── master/          # Consolidated labeled data
    │       └── TOTAL_MANUAL_LABEL_AT-TIMESTAMP_TOTAL_SAMPLE_SIZE_N.json # Master training dataset
    │
    ├── excel_exports/       # Excel export files
    │   ├── bulk_results_*.xlsx
    │   └── manual_labels_*.xlsx
    │
    └── reports/            # Project status reports
        └── progress_report_*.json
```

## Key Features

- **Progressive Few-Shot Learning:** Each corrected batch improves future predictions
- **Stable Content-Based IDs:** MD5 hashes ensure consistent identification
- **Comprehensive Evaluation:** Baseline, leave-one-out, and correction comparison tests
- **Data Separation:** Raw predictions and corrected labels stored separately
- **Modular Design:** Each function has single responsibility
- **Memory-Based Workflow:** DataFrames persist in memory, no unnecessary file I/O

## 🎯 Usage

### Basic Workflow

1. **Setup Environment**
   ```bash
   python main.py
   ```
   
2. **Data Preparation** (Option 1)
   - Load and anonymize your data files
   - Consolidate into a unified dataset
   
3. **Interactive Labeling** (Option 2)
   - Generate API predictions for unlabeled batches
   - Manually correct predictions using human expertise
   
4. **Master Label Updates** (Option 3)
   - Incorporate corrections into training dataset
   - System learns from your feedback
   
5. **Model Operations** (Option 4)
   - Fine-tune models with accumulated labeled data
   - Evaluate performance improvements
   
6. **Bulk Processing** (Option 5)
   - Apply trained models to large datasets
   - Export results for analysis

### Human-in-the-Loop Process

The system implements a continuous improvement cycle:
1. **Initial predictions** from base model
2. **Human corrections** improve training data
3. **Progressive learning** from accumulated feedback
4. **Better predictions** for future batches

## 📊 Key Features

- **🔄 Progressive Learning**: Each correction improves future predictions
- **🏷️ Human-in-the-Loop**: Combines AI efficiency with human expertise
- **🔐 Data Privacy**: Environment-based configuration keeps sensitive data secure
- **📈 Evaluation Suite**: Comprehensive testing and performance metrics
- **🚀 Bulk Processing**: Scale to thousands of records efficiently
- **🛡️ Data Anonymization**: Built-in PII protection
- **📋 Interactive Interface**: User-friendly menu system