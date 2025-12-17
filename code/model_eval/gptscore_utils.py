# gptscore_utils.py

import math
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# EVAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EVAL_MODEL_NAME = "Qwen/Qwen3-14B"

print(f"[GPTScore] Loading evaluator model: {EVAL_MODEL_NAME}")
eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_NAME)
eval_model = AutoModelForCausalLM.from_pretrained(
    EVAL_MODEL_NAME,
    device_map="auto", 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
eval_model.eval()


@dataclass
class Sample:
    transcript: str
    user_request: str
    answer: str

ASPECT_TEMPLATES: Dict[str, str] = {
    "faithfulness": """당신은 IT 프로젝트 회의록을 분석하는 전문 에이전트입니다.

[평가 목적]
아래 답변이 회의록(Transcript)에 명시적으로 포함된 정보만을 기반으로 작성되었는지 평가합니다.

[사실성(Faithfulness) / 환각 금지 규칙]
- 답변은 반드시 회의록에 명시적으로 등장하는 사실만 포함해야 합니다.
- 회의록에 없는 정보, 추론, 가정, 창작된 내용은 포함되면 안 됩니다.
- 회의록이 제공하지 않는 세부 사항을 임의로 보완해서도 안 됩니다.
- Summary JSON / Tasks JSON / Final Answer 어떤 형식이라도, 포함된 내용은 transcript 기반이어야 합니다.

[중요 – 태스크(Task) 필드 평가 규칙]
- Tasks JSON의 due 필드는 반드시 회의록에 명시적으로 등장한 표현이어야 합니다.
  (예: "이번 주 금요일까지", "내일 오전", "다음 주 초", "오늘 중")
- due_date 필드는 위 due 표현을 기반으로 시스템이 달력 날짜(YYYY-MM-DD)로 정규화한
  파생 필드(derived field)로 간주합니다.
- 따라서 due_date 자체가 회의록에 문자 그대로 등장하지 않더라도,
  해당 due 표현이 회의록에 명시적으로 존재한다면 환각으로 판단하지 않습니다.
- 단, due 표현이 회의록에 존재하지 않는데 생성된 due_date는 환각으로 간주합니다.
- due 표현으로부터 날짜를 확정할 수 없는 경우,
  due_date가 null이라면 사실성 위반이 아닙니다.

[회의록(Transcript)]
{transcript}

[사용자 요청(User Request)]
{user_request}

[모델 생성 최종 답변(Answer)]
{answer}""",

    "instruction_following": """당신은 IT 회의 분석을 수행하는 LLM 에이전트입니다.

[평가 목적]
아래 답변이 사용자 요청과 시스템 프롬프트에서 정의된 출력 규칙을 정확하게 따르고 있는지 평가합니다.

[Instruction Following 정의]
- 사용자가 Summary JSON을 요청한 경우: Summary JSON Prompt 형식을 정확히 따라야 합니다.
  - 올바른 JSON 구조와 필드명을 사용해야 합니다.
  - 내용은 회의록에 근거해야 하며, 존재하지 않는 정보를 추가하면 안 됩니다.
- 사용자가 Tasks JSON을 요청한 경우: Tasks JSON Prompt 규칙을 정확히 따라야 합니다.
  - transcript에서 실제로 파생되는 태스크들을 추출해야 합니다.
  - 태스크의 description, assignee, due, due_date 등의 필드를 규칙에 맞게 채워야 합니다.
  - 태스크는 단순 Action Item과 동일하게 취급하지 않고, transcript 기반의 실질적인 follow-up task여야 합니다.
- 사용자가 자연어 요약/이슈/ActionItem을 요청한 경우:
  - Final Answer 안에서 해당 항목들을 한국어로 명확하게 정리해야 합니다.
- 사용자가 요청한 항목(요약, 이슈, 태스크, 액션아이템 등)을 빠뜨리면 안 됩니다.
- Final Answer는 특별한 요청이 없는 한 반드시 한국어로 작성되어야 합니다.
- 회의록과 시스템 규칙을 벗어나는 형식이나 내용을 추가해서는 안 됩니다.

[회의록(Transcript)]
{transcript}

[사용자 요청(User Request)]
{user_request}

[모델 생성 최종 답변(Answer)]
{answer}""",

    "structure_clarity": """당신은 IT 프로젝트 회의 문서를 정리하는 전문 기술 작가입니다.

[평가 목적]
아래 답변이 구조적으로 명확하고, 한국어로 읽기 쉬우며, 회의 내용을 효과적으로 전달하는지 평가합니다.

[구조·명확성 정의]
- Summary, Issues, Tasks, Action Items 등이 자연스럽게 구분되어 있어야 합니다.
- 문단 또는 불릿 포인트를 활용하여 논리적으로 정리되어야 합니다.
- Final Answer는 반드시 한국어로 작성되어야 하며, 의미가 명확해야 합니다.
- 불필요하게 장황하거나 모호한 표현이 없어야 합니다.
- 회의 참석자나 PM이 내용을 빠르게 이해할 수 있도록 명료하게 작성되어야 합니다.

[회의록(Transcript)]
{transcript}

[사용자 요청(User Request)]
{user_request}

[모델 생성 최종 답변(Answer)]
{answer}""",
}


def build_gptscore_prompt(aspect: str, sample: Sample, include_answer: bool) -> str:
    template = ASPECT_TEMPLATES[aspect]
    answer_text = sample.answer if include_answer else ""
    return template.format(
        transcript=sample.transcript,
        user_request=sample.user_request,
        answer=answer_text,
    )


def _compute_answer_logprobs(
    prompt_without_answer: str,
    full_prompt: str,
) -> float:

    if not isinstance(prompt_without_answer, str):
        prompt_without_answer = str(prompt_without_answer)
    if not isinstance(full_prompt, str):
        full_prompt = str(full_prompt)

    prompt_without_answer = str(prompt_without_answer)
    full_prompt = str(full_prompt)
    prompt_without_answer = prompt_without_answer.encode("utf-8", "ignore").decode("utf-8", "ignore")
    full_prompt = full_prompt.encode("utf-8", "ignore").decode("utf-8", "ignore")


    # enc_wo = eval_tokenizer(prompt_without_answer, return_tensors="pt")
    # enc_full = eval_tokenizer(full_prompt, return_tensors="pt")
    enc_wo = eval_tokenizer(text=prompt_without_answer, return_tensors="pt")
    enc_full = eval_tokenizer(text=full_prompt, return_tensors="pt")


    input_ids_full = enc_full["input_ids"].to(eval_model.device)

    with torch.no_grad():
        outputs = eval_model(input_ids=input_ids_full)
        logits = outputs.logits 

    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)   
    target_ids = input_ids_full[:, 1:]                       

    token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  

    offset_tokens = enc_wo["input_ids"].shape[1]
    start_idx = offset_tokens - 1  # 첫 answer 토큰 예측 위치

    if start_idx >= token_logprobs.shape[1]:
        return float("nan")

    answer_token_logprobs = token_logprobs[0, start_idx:]  # [answer_len]

    if answer_token_logprobs.numel() == 0:
        return float("nan")

    # 평균 log p (길이 보정)
    return float(answer_token_logprobs.mean().item())


def compute_gptscore_for_sample(
    sample: Sample,
    aspect: str,
) -> float:
    """
    하나의 (transcript, user_request, answer)에 대해
    특정 aspect 기준 GPTScore(평균 log p)를 계산한다.
    값이 0에 가까울수록(덜 음수) 모델이 "자연스럽고 기준에 맞는 출력"이라고 보는 것.
    """
    prompt_wo_answer = build_gptscore_prompt(aspect, sample, include_answer=False)
    full_prompt = build_gptscore_prompt(aspect, sample, include_answer=True)

    score = _compute_answer_logprobs(prompt_wo_answer, full_prompt)
    return score


def compute_gptscore_batch(
    samples: List[Sample],
    aspects: List[str] = None,
) -> pd.DataFrame:
    if aspects is None:
        aspects = ["faithfulness", "instruction_following", "structure_clarity"]

    rows = []
    for sample in samples:
        row = {
            "transcript": sample.transcript,
            "user_request": sample.user_request,
            "answer": sample.answer,
        }
        for aspect in aspects:
            score = compute_gptscore_for_sample(sample, aspect)
            row[f"gptscore_{aspect}"] = score
        rows.append(row)

    return pd.DataFrame(rows)
