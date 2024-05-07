export LD_LIBRARY_PATH="/.../.../conda/envs/rag/lib"  # path to your env's lib; for JAVA and pyserini

API=together
HF_MODEL=meta-llama/Llama-2-7b-chat-hf          # model id of huggingface
TOGETHER_MODEL=meta-llama/Llama-2-7b-chat-hf    # model id of togetherai
IS_CHAT_MODEL=true
IO_INPUT_PATH=""   # path to your prompt file (JSON): a list of {"id": int, "prompt": str}
DATASTORE_ROOT=""  # where you want to save your datastore

# ====== DO IO TASK ======
python main.py  \
    --task io   \
    --api ${API}  \
    --hf_ckpt ${HF_MODEL}   \
    --together_ckpt ${TOGETHER_MODEL}   \
    --is_chat_model ${IS_CHAT_MODEL}   \
    --raw_data_dir ./raw_data/private/wiki_newest  \
    --io_input_path ${IO_INPUT_PATH}   \
    --io_output_root ./eval_data/Wikipedia/io_output   \
    --output_dir ./out \
    --datastore_root ${DATASTORE_ROOT} \

# ====== DO EVAL TASK ======
python main.py \
    --task eval   \
    --eval_input_dir ./eval_data/Wikipedia/io_output \
    --eval_output_dir ./eval_data/Wikipedia/eval_results \
    --output_dir ./out \
