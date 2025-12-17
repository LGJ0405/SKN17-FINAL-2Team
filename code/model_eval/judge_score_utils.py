# judge_score_utils.py

import os
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  

# 채점에 사용할 OpenAI 모델
JUDGE_MODEL = "gpt-4o-mini"   

@dataclass
class Sample:
    """
    LLM-as-a-judge 평가 단위
    - transcript : 회의록
    - user_request : 이 회의에 대해 모델이 수행해야 할 작업 설명
    - answer : 에이전트 최종 출력
    """
    transcript: str
    user_request: str
    answer: str

def _sanitize_text(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")

# -------------------------
# aspect별 평가 프롬프트
# -------------------------

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
  **파생 필드(derived field)** 로 간주합니다.
- 따라서 due_date 자체가 회의록에 문자 그대로 등장하지 않더라도,
  해당 due 표현이 회의록에 명시적으로 존재한다면 환각으로 판단하지 않습니다.
- 단, due 표현이 회의록에 존재하지 않는데 생성된 due_date는 환각으로 간주합니다.
- due 표현으로부터 날짜를 확정할 수 없는 경우,
  due_date가 null로 설정되어 있다면 사실성 위반이 아닙니다.

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


def build_judge_prompt(aspect: str, sample: Sample, with_reasoning: bool) -> str:
    """
    with_reasoning=True  → CoT + Final Score: X
    with_reasoning=False → 점수만 출력
    """
    base = ASPECT_TEMPLATES[aspect].format(
        transcript=sample.transcript,
        user_request=sample.user_request,
        answer=sample.answer,
    )

    if with_reasoning:
        grading_instruction = """

[채점 지시]
1) 위의 기준에 따라 모델 생성 답변(Answer)을 자세히 분석하고, 그 근거를 단계별로 서술하세요.
2) 모든 기준을 검토한 뒤, 최종 점수를 1에서 5 사이의 정수로 결정하세요.
3) 마지막 줄에는 반드시 아래 형식으로만 점수를 출력하세요.

형식:
Final Score: X

(예: Final Score: 3)
"""
    else:
        grading_instruction = """

[채점 지시]
위의 기준에 따라 모델 생성 답변(Answer)의 품질을 1에서 5 사이의 정수로 평가하세요.
- 내부적으로는 충분히 신중하게 판단하되,
- 출력은 아무 설명 없이 점수만 한 줄로 출력하세요.

형식:
X

(예: 3)
"""

    return base + grading_instruction


def compute_judge_score_for_sample(
    sample: Sample,
    aspect: str,
    model: str = JUDGE_MODEL,
    with_reasoning: bool = True,
    return_reasoning: bool = False,  # ← 추가
) -> float:
    """
    LLM-as-a-judge 방식 점수.
    - with_reasoning=True  → 체인오브쏘트 + Final Score: X
    - with_reasoning=False → 점수만 출력 (비용 절약용)
    - return_reasoning=True → (점수, 전체 응답 텍스트) 튜플로 반환
    """
    
    sample = Sample(
        transcript=_sanitize_text(sample.transcript),
        user_request=_sanitize_text(sample.user_request),
        answer=_sanitize_text(sample.answer),
    )
    
    prompt = build_judge_prompt(aspect, sample, with_reasoning=with_reasoning)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    if with_reasoning:
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else ""
        digits = "".join(ch for ch in last_line if ch.isdigit())
    else:
        digits = "".join(ch for ch in content if ch.isdigit())

    if not digits:
        score = float("nan")
    else:
        score_int = int(digits)
        score = float(score_int) if 1 <= score_int <= 5 else float("nan")

    if return_reasoning:
        return score, content

    return score


def compute_judge_score_batch(
    samples: List[Sample],
    aspects: List[str] = None,
    with_reasoning: bool = True,
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
            score = compute_judge_score_for_sample(
                sample,
                aspect,
                with_reasoning=with_reasoning,
            )
            row[f"judge_{aspect}"] = score
        rows.append(row)

    return pd.DataFrame(rows)
