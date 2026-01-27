########################################
# ========== Path Settings ==========
########################################
BASE_PATH=PATH_TO_LEGO_EVAL # "~/LEGO_Eval"
DATA_PATH=PATH_TO_PREPROCESSED_SCENES 
OBJAVERSE_ASSET_DIR=PATH_TO_OBJATHOR_ASSETS #"~/.objathor-assets/2023_09_23/assets"
MATERIAL_ASSET_DIR=PATH_TO_MATERIAL_ASSETS # "~/LEGO_Eval/floor_wall_material_image"
OBJ_IMG_DIR=PATH_TO_OBJECT_IMAGES # "~/object_images"
SAVE_DIR=PATH_TO_SAVE_DIR # "~/LEGO_Eval/logs/run1"
XORG_SCREENS=(":4" ":5") # adjust this to the screen numbers you want to use
DECOMPOSE_SAVE_DIR=PATH_TO_DECOMPOSE_SAVE_DIR # "~/LEGO_Eval/decomposed/run1" # Directory to save decomposed consraints. 

########################################
# ========== Environment Versions ==========
########################################
THOR_COMMIT_ID="3213d486cd09bcbafce33561997355983bdf8d1a"

########################################
# ========== Evaluation Settings ==========
########################################

# LLM & VLLM
MODEL_LLM=MODEL_NAME # e.g., "Qwen/Qwen3-VL-4B-Instruct"
API_KEY_LLM=API_KEY # "None" if opensource
BASE_URL_LLM=BASE_URL   # You can also use vLLM
TOP_P_LLM=TOP_P
TOP_K_LLM=TOP_K
TEMPERATURE_LLM=TEMPERATURE

MODEL_VLLM=MODEL_NAME # e.g., "Qwen/Qwen3-VL-4B-Instruct"
API_KEY_VLLM=API_KEY # "None" if opensource
BASE_URL_VLLM=BASE_URL # You can also use vLLM
TOP_P_VLLM=TOP_P
TOP_K_VLLM=TOP_K
TEMPERATURE_VLLM=TEMPERATURE

# Constraint Identification
MODEL_CONST=MODEL_NAME # e.g., "Qwen/Qwen3-VL-4B-Instruct"
API_KEY_CONST=API_KEY # "None" if opensource
BASE_URL_CONST=BASE_URL # You can also use vLLM

########################################
# ========== Runtime Parameters ==========
########################################
BATCH_SIZE=BATCH_SIZE_for_EVALUATION
MAX_INST_WORKERS=WORKER_NUM #Number of maximum parallel instruction processes
MAX_TOOL_WORKERS=WORKER_NUM # Number of maximum parallel tool execution processes

python evaluation/main.py \
    --base_path $BASE_PATH \
    --data_path $DATA_PATH \
    --objaverse_dir $OBJAVERSE_ASSET_DIR \
    --material_dir $MATERIAL_ASSET_DIR \
    --obj_img_dir $OBJ_IMG_DIR \
    --save_dir $SAVE_DIR \
    --thor_id $THOR_COMMIT_ID \
    --api_key_llm $API_KEY_LLM \
    --api_key_vllm $API_KEY_VLLM \
    --api_key_const $API_KEY_CONST \
    --model_llm $MODEL_LLM \
    --model_vllm $MODEL_VLLM \
    --model_const $MODEL_CONST \
    --base_url_llm $BASE_URL_LLM \
    --base_url_vllm $BASE_URL_VLLM \
    --base_url_const $BASE_URL_CONST \
    --top_p_llm $TOP_P_LLM \
    --top_k_llm $TOP_K_LLM \
    --temperature_llm $TEMPERATURE_LLM \
    --top_p_vllm $TOP_P_VLLM \
    --top_k_vllm $TOP_K_VLLM \
    --temperature_vllm $TEMPERATURE_VLLM \
    --batch_size $BATCH_SIZE \
    --max_inst_workers $MAX_INST_WORKERS \
    --max_tool_workers $MAX_TOOL_WORKERS \
    --xorg_screens "${XORG_SCREENS[@]}" \
    --decompose_save_dir $DECOMPOSE_SAVE_DIR \
    # --lego_bench \

# Remove --lego_bench if you are not testing on LEGO-Bench
