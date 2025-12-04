import json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from huggingface_hub import login
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()
login(token=os.getenv('HF_TOKEN'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=True,
    torch_dtype="auto",
    device_map="auto"
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4bit 학습 준비
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# LoRA 적용
model = get_peft_model(model, lora_config)

# gradient checkpointing
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.print_trainable_parameters()

MAX_LEN = 256
ANSWER_MAX = 128

def tokenize_function(example):
    messages = example["messages"]
    assert messages[-1]["role"] == "assistant"

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_text = messages[-1]["content"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # 1) 프롬프트 길이 제한: 너무 길면 뒤쪽 ANSWER_MAX 만큼만 남김
    max_prompt_len = MAX_LEN - ANSWER_MAX
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]  # 뒤쪽만 유지

    # 2) 답변 길이 제한: 남은 자리에 맞춰 자르기
    max_answer_len = MAX_LEN - len(prompt_ids)
    if max_answer_len <= 0:
        answer_ids = []
    else:
        answer_ids = answer_ids[:max_answer_len]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)

    pad_len = MAX_LEN - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

    labels = [-100] * len(prompt_ids) + answer_ids + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# 데이터셋 로드
dataset = load_dataset("json", data_files={"train": ["syn_data.jsonl"]})

print(dataset["train"][0])

dataset = dataset["train"].train_test_split(
    test_size=0.1,        # 10% = validation
    shuffle=True,         # 섞기 (중요)
    seed=42               # 재현성을 위해 고정
)

train_ds = dataset["train"]
eval_ds = dataset["test"]

tokenized_train = train_ds.map(
    tokenize_function,
    batched=False,
    remove_columns=train_ds.column_names
)

tokenized_eval = eval_ds.map(
    tokenize_function,
    batched=False,
    remove_columns=eval_ds.column_names
)

sample = tokenized_train[0]
print("input_ids:", sample["input_ids"][:80])
print("labels   :", sample["labels"][:80])

valid_labels = sum(1 for x in sample["labels"] if x != -100)
print("유효 라벨 토큰 개수:", valid_labels)

from transformers import TrainingArguments, Trainer, default_data_collator

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,

    bf16=True,          # A40이면 OK
    fp16=False,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    logging_dir="./logs",
    logging_steps=30,
    logging_strategy="steps",
    logging_first_step=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,

    optim="adamw_torch",
    ddp_find_unused_parameters=False,

    remove_unused_columns=False,   # ★ labels 컬럼 날리지 말 것
    push_to_hub=False,             # HF 로그인 요구 안 하게
    report_to=["none"],            # 텐서보드/허브 로그 끔 (원하면 바꿔도 됨)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=default_data_collator,
    tokenizer=tokenizer,           # ★ 있으면 internal padding 등 깔끔
)

trainer.train()

# 학습 끝난 후 모델 저장
trainer.save_model("./Qwen2.5_3B_trained_model_v1")

# 토크나이저도 같이 저장 (나중에 로드할 때 필요)
tokenizer.save_pretrained("./Qwen2.5_3B_trained_model_v1")

from huggingface_hub import HfApi
api = HfApi()
api.create_repo(
    repo_id="poketmon/Qwen2.5_3B_trained_model_v1",
    repo_type="model",
    private=False,
    exist_ok=True,
)
api.upload_folder(
    folder_path="./Qwen2.5_3B_trained_model_v1",
    repo_id="poketmon/Qwen2.5_3B_trained_model_v1",
    repo_type="model",
    commit_message="Upload LoRA adapter v1",
)


