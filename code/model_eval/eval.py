import json
import time
from datetime import datetime

from main_model import preprocess_transcript
from sllm_tool_binding import agent_main

# from gptscore_utils import Sample as GPTSample, compute_gptscore_for_sample
# from judge_score_utils import Sample as JudgeSample, compute_judge_score_for_sample


DEFAULT_USER_REQUEST = "이 회의록을 기반으로 요약, 이슈, 후속 태스크를 한국어로 정리하라."
DEFAULT_ASPECTS = ["faithfulness", "instruction_following", "structure_clarity"]


def make_answer_text(result: dict) -> str:
    agendas_text = json.dumps({"agendas": result.get("agendas", [])}, ensure_ascii=False, indent=2)
    tasks_text = json.dumps({"tasks": result.get("tasks", [])}, ensure_ascii=False, indent=2)

    return " [Summary] \n" + agendas_text + "\n\n [Tasks] \n" + tasks_text

def run_eval(transcript: str, result: dict, user_request: str = DEFAULT_USER_REQUEST) -> dict:
    from gptscore_utils import Sample as GPTSample, compute_gptscore_for_sample
    from judge_score_utils import Sample as JudgeSample, compute_judge_score_for_sample

    answer_text = make_answer_text(result)

    # GPTScore
    gpt_sample = GPTSample(transcript=transcript, user_request=user_request, answer=answer_text)
    gptscore = {a: compute_gptscore_for_sample(gpt_sample, a) for a in DEFAULT_ASPECTS}

    # LLM-as-a-judge (reasoning 포함)
    judge_sample = JudgeSample(transcript=transcript, user_request=user_request, answer=answer_text)
    judge = {}
    for a in DEFAULT_ASPECTS:
        score, reasoning = compute_judge_score_for_sample(
            judge_sample,
            a,
            with_reasoning=True,
            return_reasoning=True,
        )
        judge[a] = {"score": score, "reasoning": reasoning}

    return {"gptscore": gptscore, "judge": judge}


if __name__ == "__main__":
    domain_input = input("\n검색할 도메인을 입력하세요 (예: IT, 의료, 법률, 엔터없이 입력 시 전체 검색): ").strip()
    raw_transcript = input("\n회의록 전문을 입력하세요:\n")

    t0 = time.perf_counter()    # 시간 측정 시작

    transcript = preprocess_transcript(raw_transcript)  

    # 1) 파이프라인 결과 생성
    result = agent_main(domain_input, transcript)  

    t1 = time.perf_counter()

    # 2) 평가 수행 
    eval_result = run_eval(transcript, result)

    output = {
        "meta": {
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "domain_input": domain_input,
            "aspects": DEFAULT_ASPECTS,
            "user_request": DEFAULT_USER_REQUEST,
            "sec_until_eval": round(t1 - t0, 3),
            "raw_char_len": len(raw_transcript),
            "preprocessed_char_len": len(transcript),
        },
        "result": result,
        "eval": eval_result,
    }

    print("\n" + "=" * 50)
    print("최종 결과 + 평가")
    print("=" * 50)
    print(json.dumps(output, ensure_ascii=False, indent=2))