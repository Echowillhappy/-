# config.py

# --- Model Configuration ---
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat" # or Qwen2-0.5B-Instruct

# --- PEFT/LoRA Configuration ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# For Qwen1.5: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# For Qwen2 (0.5B/1.5B/7B): ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Quantization (QLoRA) ---
USE_4BIT_QUANTIZATION = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"

# --- Training Arguments ---
OUTPUT_DIR = "./results_quad_extraction"
TRAIN_BATCH_SIZE = 2 # Adjust based on VRAM
GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 10
SAVE_STEPS_RATIO = 0.25 # Save every 25% of total steps, or set specific save_steps
MAX_SEQ_LENGTH = 1024

# --- Data Files & Splitting ---
TRAIN_FILE = "train.json" # Your main training data
VALIDATION_SPLIT_RATIO = 0.1 # 10% of train.json will be used for validation
# TEST_INPUT_FILE will be for the data you want to predict on without ground truth
TEST_INPUT_FILE = "test1.json" # Your test file for inference

# --- Output/Plotting ---
# Predictions for TEST_INPUT_FILE (e.g., test1.json)
TEST_PREDICTIONS_FILE = f"{OUTPUT_DIR}/test1_predictions.txt"
# Predictions for the validation split (for internal evaluation)
VALIDATION_PREDICTIONS_FILE = f"{OUTPUT_DIR}/validation_split_predictions.txt"

TRAINING_PLOTS_DIR = f"{OUTPUT_DIR}/training_plots"
VALIDATION_EVAL_PLOTS_DIR = f"{OUTPUT_DIR}/validation_evaluation_plots" # For F1 scores on validation set