echo "run demo"
# export TRAIN_FILE_DIR='/mnt/data/liyihao/workspace/llm_train_translate/test'
# export TARGET_MODEL_DIR='/tmp/workspace/train/save_sft'
# export BASE_MODEL_DIR='/mnt/data/liyihao/model_zoo/model-qwen2-5-0.5b-instruct'
# deactivate since the competition do not support env var declaration
# if [ -z "$do_pt" ]; then
#     echo "only sft"
#     cd /tmp/workspace/train && python prepare_v2.py
#     cd /tmp/workspace/train && llamafactory-cli train train.yaml
# else
#     echo "pt and sft"
#     cd /tmp/workspace/train && python prepare_v2.py --current_stage 0 --do_pt
#     cd /tmp/workspace/train && llamafactory-cli train train.yaml
#     cd /tmp/workspace/train && python prepare_v2.py --current_stage 1 --do_pt
#     cd /tmp/workspace/train && llamafactory-cli train train.yaml
# fi
# echo "only sft"
# cd /tmp/workspace/train && python prepare_v2.py
# cd /tmp/workspace/train && llamafactory-cli train train.yaml
# echo "pt and sft"
cd /tmp/workspace/train && python prepare_v2.py --current_stage 0 --do_pt
cd /tmp/workspace/train && llamafactory-cli train train.yaml
cd /tmp/workspace/train && python prepare_v2.py --current_stage 1 --do_pt
cd /tmp/workspace/train && llamafactory-cli train train.yaml