import os
import sys
import time
import math
import argparse
import json
import yaml
import gc
import logging
import pandas as pd
from collections import Counter
from copy import deepcopy
from tqdm import tqdm, trange
from transformers import AutoTokenizer
try:
    from .darvin_dataset import load_dataframe
except Exception:  # executed as script
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(__file__))
    from darvin_dataset import load_dataframe

# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.INFO)
# # 创建控制台处理器
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# # 创建日志格式
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# # 添加处理器到日志记录器
# logger.addHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LANGUAGE_MAPPING = {
    "zh": "Chinese",
    "en": "English",
    "ru": "Russian",
    "ar": "Arabic",
    "tr": "Turkish",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "he": "Hebrew",
    "fa": "Farsi",
    "ja": "Japanese",
    "fr": "French",
    "th": "Thai",
    "el": "Greek",
    "vi": "Vietnamese",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    # "sv": "Swedish",
    # "hu": "Hungarian",
    # "cs": "Czech",
    # "no": "Norwegian",
    # "fi": "Finnish",
    # "da": "Danish",
    # "bg": "Bulgarian",
    # "ko": "Korean",
    # "hi": "Hindi",  
}
used_languages = ['zh', 'en', 'ru', 'fr', 'ar', 'pt', 'es', 'vi']
used_languages_mapper = [
    # 'zh2en', 'zh2ru', 'zh2fr', 'zh2ar', 'zh2pt', 'zh2es', 
    'en2zh', 'en2ru', 'en2fr', 'en2ar', 'en2pt', 'en2es', 'en2vi',
]
# used_languages = ['zh', 'en', 'ru', 'fr', 'ar', 'pt', 'es', 'pl']
# used_languages_mapper = [
#     # 'zh2en', 'zh2ru', 'zh2fr', 'zh2ar', 'zh2pt', 'zh2es', 
#     'en2zh', 'en2ru', 'en2fr', 'en2ar', 'en2pt', 'en2es', 'en2pl'
# ]
sample_dict = {k:0 for k in used_languages_mapper}

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--do_pt', default=False, action='store_true')
    parse.add_argument('--current_stage', default=0, type=int)
    args = parse.parse_args()
    return args

args = parse_args()
do_pt = args.do_pt
# use_lora = args.use_lora
current_stage = args.current_stage
assert current_stage in [0, 1]
# MAX_FILE_SIZE = 3072 # MB
MAX_TOKEN_SIZE = 256 # 512 # for larger batch size
MAX_SAMPLE_SIZE = 600000 # 500000 # 1000000 # 3600000
# if (do_pt) and (current_stage==0):
#     MAX_SAMPLE_SIZE = MAX_SAMPLE_SIZE * 2
MAX_LANG_SAMPLE_SIZE = MAX_SAMPLE_SIZE // len(used_languages_mapper)
PER_BATCH_SIZE = 16 # 32
ACC_STEPS = 8
base_model_dir = os.getenv("BASE_MODEL_DIR") or "/tmp/workspace/source_model"
# New: Darvin dataset sources — prefer bundle path, else dataset dir (extracted bundle or zips directory)
train_bundle_path = os.getenv("DARVIN_TRAIN_BUNDLE_PATH")
train_dataset_dir = os.getenv("DARVIN_DATASET_DIR") or os.getenv("TRAIN_FILE_DIR") or "/tmp/ds"
# adapter_path = "/tmp/adapter"
# sft_path = '/tmp/workspace/train/full_sft'
mid_model_dir = "/tmp/workspace/target_model"
target_model_dir = os.getenv("TARGET_MODEL_DIR") or "/tmp/workspace/target_model"
train_yaml_filepath = "/tmp/workspace/train/train.yaml"
# merge_yaml_filepath = "/tmp/workspace/train/merge.yaml"
dataset_info_filepath = "/tmp/workspace/train/data/dataset_info.json"
logging.info("base_model_dir: {}".format(base_model_dir))
logging.info("train_bundle_path: {}".format(train_bundle_path))
logging.info("train_dataset_dir: {}".format(train_dataset_dir))
logging.info("target_model_dir: {}".format(target_model_dir))
logging.info("train_yaml_filepath: {}".format(train_yaml_filepath))
# logging.info("merge_yaml_filepath: {}".format(merge_yaml_filepath))

time_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

# ---load Darvin datasets (ZIP with dataset_schema.json + annotations.jsonl)---
try:
    if train_bundle_path:
        df = load_dataframe(bundle_path=train_bundle_path)
    else:
        df = load_dataframe(dataset_dir=train_dataset_dir)
except Exception as e:
    logging.error(f"Failed to load Darvin dataset(s): {e}")
    raise

# filter and sample by languages and token length budget
df = df.dropna()
df = df[df['from'].isin(used_languages)]
df = df[df['to'].isin(used_languages)]
df['a2b'] = df.apply(lambda x: '{}2{}'.format(x['from'], x['to']), axis=1)
df['b2a'] = df.apply(lambda x: '{}2{}'.format(x['to'], x['from']), axis=1)
df['used_tag'] = df.apply(lambda x: (x['a2b'] in used_languages_mapper) or (x['b2a'] in used_languages_mapper), axis=1)
df = df[df['used_tag'] == True]

dfs = []
for _, group in df.groupby(['from', 'to'], sort=False):
    tmp = group.copy()
    if len(tmp) == 0:
        continue
    ts = []
    ulms = list(set(tmp['a2b'].tolist() + tmp['b2a'].tolist()) & set(used_languages_mapper))
    loop_tag = any(sample_dict[ulm] < MAX_LANG_SAMPLE_SIZE for ulm in ulms)
    if not loop_tag:
        continue
    for ulm in ulms:
        if sample_dict[ulm] >= MAX_LANG_SAMPLE_SIZE:
            continue
        t = pd.concat([tmp[tmp['a2b'] == ulm], tmp[tmp['b2a'] == ulm]])
        t['concated'] = t.apply(lambda x: '\n'.join([str(x['text']), str(x['translated'])]), axis=1)
        t['token_num'] = t['concated'].map(lambda x: len(tokenizer(x)['input_ids']))
        t = t[t['token_num'] <= MAX_TOKEN_SIZE]
        t_num = min(len(t), MAX_LANG_SAMPLE_SIZE - sample_dict[ulm])
        if t_num <= 0:
            continue
        t = t.sample(n=t_num)
        ts.append(t)
        sample_dict[ulm] += t_num
    if ts:
        tmp2 = pd.concat(ts)
        tmp2 = tmp2.drop(columns=['concated', 'token_num'], errors='ignore')
        dfs.append(tmp2)
        gc.collect()

if not dfs:
    raise ValueError("no_samples_after_language_and_token_filters")
df = pd.concat(dfs)
df = df.drop_duplicates(keep='first', ignore_index=True)
df = df.astype(str)
logging.info('source sample num: {}'.format(len(df)))
logging.info('source sample memory size: {}MB'.format(sys.getsizeof(df)/1024/1024))
# logging.info('language counter: {}'.format(Counter(df['from'].tolist() + df['to'].tolist())))
logging.info('language trans counter: {}'.format(sample_dict))
logging.info('Finish reading csv time costs: {}s'.format(time.time() - time_start))
del dfs, tokenizer
gc.collect()
df2 = pd.DataFrame()
df2['from'] = df['to']
df2['to'] = df['from']
df2['text'] = df['translated']
df2['translated'] = df['text']
df = pd.concat([df, df2], ignore_index=True)
df = df.drop_duplicates(keep='first', ignore_index=True)
del df2
gc.collect()
logging.info('augmented sample num: {}'.format(len(df)))
logging.info('augmented sample memory size: {}MB'.format(sys.getsizeof(df)/1024/1024))
logging.info('Finish augmentation time costs: {}s'.format(time.time() - time_start))
sample_counter = len(df)
# ---save train file---
res_list = []
# prompt_template = "请将以下文字从{from}翻译为{to}，只返回译文即可，无需其他任何信息:\n{text}"
pt_template = "Please translate the following text from \"{from}\" to \"{to}\", return only the translated text, no additional information: \"{text}\" {translated}"
sft_template = "Please translate the following text from \"{from}\" to \"{to}\", return only the translated text, no additional information: \"{text}\""
for idx, row in df.iterrows():
    s, t, st, tt = row['from'], row['to'], row['text'].strip(), row['translated'].strip()
    if (do_pt) and (current_stage==0):
        # pt/sft format
        tmp_text = pt_template.replace('{from}',s).replace('{to}',t).replace('{text}',st).replace('{translated}',tt)
        res_list.append({"text": tmp_text})
    else:
        # sharegpt sft format
        tmp_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": sft_template.replace('{from}',s).replace('{to}',t).replace('{text}',st)
                },
                {
                    "role": "assistant",
                    "content": tt
                }
            ]
        }
        res_list.append(tmp_dict)
logging.info('processed dataset sample num: {}'.format(len(res_list)))
logging.info('processed dataset memory size: {}MB'.format(sys.getsizeof(res_list)/1024/1024))
logging.info('Finish processing dataset time costs: {}s'.format(time.time() - time_start))


# ---dataset_info---
dataset_info = dict()
if (do_pt) and (current_stage==0):
    translate_data_filename = 'tmp_translate_data_pt.json'
    with open(translate_data_filename, 'w', encoding="utf-8") as f:
        json.dump(res_list, f, indent=2, ensure_ascii=False)
    dataset_info[translate_data_filename[:-5]] = \
    {
        "file_name": os.path.join(os.getcwd(), translate_data_filename),
        "columns": {
            "prompt": "text"
        }
    }
else:
    translate_data_filename = 'tmp_translate_data_sft.json'
    with open(translate_data_filename, 'w', encoding="utf-8") as f:
        json.dump(res_list, f, indent=2, ensure_ascii=False)
    dataset_info[translate_data_filename[:-5]] = \
    {
        "file_name": os.path.join(os.getcwd(), translate_data_filename),
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
os.makedirs(os.path.dirname(dataset_info_filepath), exist_ok=True)
with open(dataset_info_filepath, "w") as f:
    f.write(json.dumps(dataset_info, indent=2, ensure_ascii=False))

def dump_yaml(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, indent=2, allow_unicode=True, sort_keys=False, line_break="\n")

# ---train_config---
if (do_pt) and (current_stage==0):
    epoch = 1 # 2 # 3
    per_batch_size = PER_BATCH_SIZE
    # sample_counter = max(sample_counter,1e6) # max(sample_counter,1e6)
    # sample_counter = 3200
    eval_step = int(sample_counter*epoch/per_batch_size/ACC_STEPS/2)
    train_config = {
        "model_name_or_path": base_model_dir,

        "stage": "pt",
        "do_train": True,
        "finetuning_type": "full",
        # deepspeed: ./ds_z3_config.json

        "dataset": translate_data_filename[:-5],
        "template": "qwen",
        "cutoff_len": 512,
        "max_samples": sample_counter,
        # "overwrite_cache": True,
        # "use_cache": False,
        # "cache_dir": "/tmp/workspace/cache", # None,
        "preprocessing_num_workers": 4, # -1,
        "preprocessing_batch_size": 256,

        "output_dir": mid_model_dir, # sft_path,
        "logging_steps": eval_step//100,
        "save_steps": -1, # eval_step,
        "save_only_model": True,
        "plot_loss": True,
        "overwrite_output_dir": True,

        "per_device_train_batch_size": per_batch_size,
        "gradient_accumulation_steps": ACC_STEPS,
        "learning_rate": 4e-4, # 0.0001,
        "weight_decay": 0.1,
        "num_train_epochs": epoch,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,

        "val_size": 0.0,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "no",
        "eval_steps": eval_step
    }

    dump_yaml(train_yaml_filepath, train_config)
elif (do_pt) and (current_stage==1):
    epoch = 1 # 2 # 3
    per_batch_size = PER_BATCH_SIZE
    # sample_counter = max(sample_counter,1e6) # max(sample_counter,1e6)
    # sample_counter = 3200
    eval_step = int(sample_counter*epoch/per_batch_size/ACC_STEPS/2)
    train_config = {
        "model_name_or_path": mid_model_dir,

        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        # deepspeed: ./ds_z3_config.json

        "dataset": translate_data_filename[:-5],
        "template": "qwen",
        "cutoff_len": 512,
        "max_samples": sample_counter,
        # "overwrite_cache": True,
        # "use_cache": False,
        # "cache_dir": "/tmp/workspace/cache", # None,
        "preprocessing_num_workers": 4, # -1,
        "preprocessing_batch_size": 256,

        "output_dir": target_model_dir, # sft_path,
        "logging_steps": eval_step//100,
        "save_steps": -1, # eval_step,
        "save_only_model": True,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": per_batch_size,
        "gradient_accumulation_steps": ACC_STEPS,
        "learning_rate": 4e-4, # 0.0001,
        "weight_decay": 0.1,
        "num_train_epochs": epoch,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,

        "val_size": 0.0,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "no",
        "eval_steps": eval_step
    }

    dump_yaml(train_yaml_filepath, train_config)
else:
    epoch = 2 # 3
    per_batch_size = PER_BATCH_SIZE
    # sample_counter = max(sample_counter,1e6) # max(sample_counter,1e6)
    # sample_counter = 3200
    eval_step = int(sample_counter*epoch/per_batch_size/ACC_STEPS/2)
    train_config = {
        "model_name_or_path": base_model_dir,

        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        # deepspeed: ./ds_z3_config.json

        "dataset": translate_data_filename[:-5],
        "template": "qwen",
        "cutoff_len": 512,
        "max_samples": sample_counter,
        # "overwrite_cache": True,
        # "use_cache": False,
        # "cache_dir": "/tmp/workspace/cache", # None,
        "preprocessing_num_workers": 4, # -1,
        "preprocessing_batch_size": 256,

        "output_dir": target_model_dir, # sft_path,
        "logging_steps": eval_step//100,
        "save_steps": -1, # eval_step,
        "save_only_model": True,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": per_batch_size,
        "gradient_accumulation_steps": ACC_STEPS,
        "learning_rate": 4e-4, # 0.0001,
        "weight_decay": 0.1,
        "num_train_epochs": epoch,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,

        "val_size": 0.0,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "no",
        "eval_steps": eval_step
    }

    dump_yaml(train_yaml_filepath, train_config)

# merge_config = {
#     "model_name_or_path": base_model_dir,
#     "adapter_name_or_path": adapter_path,
#     "template": "qwen",
#     "finetuning_type": "lora",
#     "export_dir": target_model_dir
# }

# dump_yaml(merge_yaml_filepath, merge_config)

logging.info('Finish saving dataset_info and config time costs: {}s'.format(time.time() - time_start))
