import json
import re
import hashlib
import statistics
from typing import List, Dict, Any, Set

from jsonschema import validate, ValidationError
from sentence_transformers import SentenceTransformer, util

###############################################
# 0. 스키마 정의 (who, when 모두 null 허용)
###############################################

ASSISTANT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "agendas": {
            "type": "array",
            "items": {"type": "string"}
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "who":  {"type": ["string", "null"]},
                    "what": {"type": "string"},
                    "when": {"type": ["string", "null"]}
                },
                "required": ["who", "what", "when"]
            }
        }
    },
    "required": ["agendas", "tasks"]
}

BERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
bert_model = SentenceTransformer(BERT_MODEL_NAME)

###############################################
# 1. JSON 스키마 검증
###############################################

def validate_schema(assistant_json_str: str):
    try:
        data = json.loads(assistant_json_str)
        validate(instance=data, schema=ASSISTANT_OUTPUT_SCHEMA)
        return True, data
    except (json.JSONDecodeError, ValidationError):
        return False, None

###############################################
# 2. 화자 / 담당자 처리
###############################################

def extract_speakers(transcript: str) -> Set[str]:
    """
    '화자:' 형태로 나타나는 이름/역할을 추출.
    예: 'PM:', '개발자:', 'QA 엔지니어:'
    """
    speakers = set()
    for line in transcript.splitlines():
        line = line.strip()
        m = re.match(r'^(.+?):', line)
        if m:
            name = m.group(1).strip()
            speakers.add(name)
    return speakers


def normalize_name(name: str) -> str:
    """
    비교용 이름 정규화:
    - 앞뒤 공백 제거
    - 중간 공백 제거 (예: 'QA 엔지니어' == 'QA엔지니어')
    """
    s = name.strip()
    s = re.sub(r"\s+", "", s)
    return s


def check_unknown_who(tasks: List[Dict], speakers: Set[str]):
    """
    unknown_who 판정:
    - who == None 이면 '담당자 미정' → 정상
    - 쉼표(,)나 슬래시(/)로 여러 명이 들어간 경우: 각 부분을 분리해서
      하나라도 화자 목록에 매칭되면 정상 처리
    - 그 외 단일 문자열은 normalize 후 화자 목록과 매칭
    """
    errors = []

    # 화자를 정규화한 세트로 준비
    norm_speakers = {normalize_name(s) for s in speakers}

    for i, t in enumerate(tasks):
        who = t["who"]

        # null이면 '미정' 담당자 → 정상으로 본다
        if who is None:
            continue

        raw = who

        # 1) 쉼표/슬래시 기준으로 여러 명 분리
        parts = [p.strip() for p in re.split(r'[,/]', raw) if p.strip()]

        # 2) 분리된 게 없다면 그대로 하나로 취급
        if not parts:
            parts = [raw]

        # 3) 각 part 중 하나라도 화자에 매칭되면 OK
        ok = False
        for p in parts:
            n = normalize_name(p)
            if n in norm_speakers:
                ok = True
                break
            # 조금 더 관대하게: 부분 문자열 매칭도 허용
            if any(n in s or s in n for s in norm_speakers):
                ok = True
                break

        if not ok:
            errors.append({"index": i, "who": who})

    return errors

###############################################
# 3. 기한(when) 문자열 존재 여부 검사
###############################################

DEADLINE_PATTERNS = [
    r'\d+월\s*\d+일',
    r'\d+일까지',
    r'내일[^\s]*',
    r'오늘[^\s]*',
    r'이번\s*주[^\s]*',
    r'다음\s*주[^\s]*',
    r'다음\s*회의[^\s]*',
]

def extract_deadline_phrases(transcript: str) -> Set[str]:
    phrases = set()
    for pat in DEADLINE_PATTERNS:
        for m in re.finditer(pat, transcript):
            phrases.add(m.group(0))
    return phrases


def check_when_presence(tasks: List[Dict], transcript: str):
    """
    when이 None이면 '기한 미정'으로 보고 정상 처리.
    문자열인 when만 회의록/패턴에 존재하는지 검사.
    """
    found_phrases = extract_deadline_phrases(transcript)
    errors = []
    for i, t in enumerate(tasks):
        w = t.get("when")

        if w is None:
            continue

        if (w not in transcript) and (w not in found_phrases):
            errors.append({"index": i, "when": w})
    return errors

###############################################
# 4. 의미 기반 유사도(BERT cosine similarity)
###############################################

def check_task_semantic_similarity(tasks: List[Dict], transcript: str, threshold=0.5):
    """
    각 task.what과 transcript 전체의 BERT cosine similarity를 계산하고,
    threshold보다 낮으면 low_similarity로 기록.
    """
    errors = []

    emb_transcript = bert_model.encode(transcript, convert_to_tensor=True)

    for i, t in enumerate(tasks):
        what = t["what"]
        if not isinstance(what, str) or not what.strip():
            continue

        emb_what = bert_model.encode(what, convert_to_tensor=True)
        sim = util.cos_sim(emb_what, emb_transcript).item()

        if sim < threshold:
            errors.append({
                "index": i,
                "what": what,
                "similarity": float(sim)
            })

    return errors

###############################################
# 5. 중복 샘플 검사(hash)
###############################################

def hash_transcript(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

###############################################
# 6. 개별 샘플 검증
###############################################

def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    messages = sample["messages"]

    user_msg = messages[1]["content"]
    assistant_msg = messages[-1]["content"]

    schema_ok, data = validate_schema(assistant_msg)

    if not schema_ok:
        return {
            "schema_ok": False,
            "score": 0.0,
            "issues": {"schema_error": True}
        }

    tasks = data["tasks"]

    speakers = extract_speakers(user_msg)
    unknown_who = check_unknown_who(tasks, speakers)
    when_errors = check_when_presence(tasks, user_msg)
    semantic_low = check_task_semantic_similarity(tasks, user_msg, threshold=0.5)

    score = 1.0

    if unknown_who:
        score -= 0.2

    if when_errors:
        score -= 0.2

    if semantic_low:
        total_tasks = max(len(tasks), 1)
        ratio = len(semantic_low) / total_tasks
        score -= 0.2 * ratio

    score = max(score, 0.0)

    return {
        "schema_ok": True,
        "score": score,
        "issues": {
            "unknown_who": unknown_who,
            "when_missing": when_errors,
            "low_similarity": semantic_low
        }
    }

###############################################
# 7. JSONL 파일 전체 처리
###############################################

def validate_jsonl(path: str):
    results = []
    seen_hashes = {}
    duplicates = []

    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            transcript = sample["messages"][1]["content"]

            h = hash_transcript(transcript)
            if h in seen_hashes:
                duplicates.append((seen_hashes[h], idx))
            else:
                seen_hashes[h] = idx

            result = validate_sample(sample)
            result["index"] = idx
            results.append(result)

    return results, duplicates

###############################################
# 8. 결과 요약 리포트
###############################################

def summarize_results(results, duplicates, top_n: int = 20):
    total = len(results)
    print("===== 요약 리포트 =====")
    print(f"총 샘플 수: {total}")
    print()

    # 1. 스키마 통계
    schema_ok_count = sum(1 for r in results if r.get("schema_ok"))
    schema_bad_count = total - schema_ok_count
    print("[스키마 검사]")
    print(f"  schema_ok=True  : {schema_ok_count}")
    print(f"  schema_ok=False : {schema_bad_count}")
    print()

    # 2. score 통계
    scores = [r["score"] for r in results if "score" in r]
    if scores:
        avg_score = statistics.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        print("[score 통계]")
        print(f"  평균 score : {avg_score:.4f}")
        print(f"  최소 score : {min_score:.4f}")
        print(f"  최대 score : {max_score:.4f}")

        bins = [
            (0.9, 1.01, "0.9 ~ 1.0"),
            (0.8, 0.9,  "0.8 ~ 0.9"),
            (0.7, 0.8,  "0.7 ~ 0.8"),
            (0.5, 0.7,  "0.5 ~ 0.7"),
            (0.0, 0.5,  "0.0 ~ 0.5"),
        ]

        print("  score 분포:")
        for lo, hi, label in bins:
            cnt = sum(1 for s in scores if lo <= s < hi)
            print(f"    {label}: {cnt}")
    print()

    # 3. 이슈 통계
    print("[이슈 통계]")

    samples_unknown = []
    total_unknown_entries = 0
    samples_when = []
    total_when_entries = 0
    samples_low_sim = []
    total_low_sim_entries = 0

    for r in results:
        issues = r.get("issues") or {}

        uw = issues.get("unknown_who") or []
        wm = issues.get("when_missing") or []
        ls = issues.get("low_similarity") or []

        if uw:
            samples_unknown.append(r)
            total_unknown_entries += len(uw)

        if wm:
            samples_when.append(r)
            total_when_entries += len(wm)

        if ls:
            samples_low_sim.append(r)
            total_low_sim_entries += len(ls)

    print(f"  unknown_who 있는 샘플 수 : {len(samples_unknown)}")
    print(f"  unknown_who 총 개수     : {total_unknown_entries}")
    print(f"  when_missing 있는 샘플 수: {len(samples_when)}")
    print(f"  when_missing 총 개수    : {total_when_entries}")
    print(f"  low_similarity 있는 샘플 수: {len(samples_low_sim)}")
    print(f"  low_similarity 총 개수    : {total_low_sim_entries}")
    print()

    # 4. 중복 샘플 정보
    print("[중복 샘플]")
    print(f"  중복 쌍 개수: {len(duplicates)}")
    if duplicates:
        print("  (원 인덱스, 중복 인덱스) 예시 상위 20개:")
        for pair in duplicates[:20]:
            print(f"    {pair}")
    print()

    # 5. score 낮은 샘플 상위 N개
    print(f"[score 낮은 샘플 상위 {top_n}개]")
    sorted_results = sorted(results, key=lambda r: r.get("score", 0.0))
    for r in sorted_results[:top_n]:
        idx = r.get("index")
        score = r.get("score", 0.0)
        issues = r.get("issues") or {}
        uw = issues.get("unknown_who") or []
        wm = issues.get("when_missing") or []
        ls = issues.get("low_similarity") or []

        print("-" * 60)
        print(f"index: {idx}, score: {score:.4f}")
        print(f"  unknown_who 개수    : {len(uw)}")
        print(f"  when_missing 개수  : {len(wm)}")
        print(f"  low_similarity 개수: {len(ls)}")

        if ls:
            preview_n = min(2, len(ls))
            print(f"  low_similarity 예시 {preview_n}개:")
            for item in ls[:preview_n]:
                print(
                    f"    - task_idx={item['index']}, "
                    f"sim={item['similarity']:.4f}, "
                    f"what={item['what']}"
                )

    print()
    print("===== 요약 끝 =====")


###############################################
# 실행 예시
###############################################

if __name__ == "__main__":
    filename = "syn_data_dc.jsonl"  # 여기 경로만 필요에 따라 바꾸면 됨
    results, duplicates = validate_jsonl(filename)
    summarize_results(results, duplicates, top_n=20)

#############################
    # -------------------------------------------------------
    # score < 0.5 제거 후 새 jsonl 저장
    # -------------------------------------------------------
    threshold = 0.5
    bad_idx = {r["index"] for r in results if r.get("score", 0.0) < threshold}

    print(f"\n[INFO] score < {threshold} 인 샘플 수: {len(bad_idx)}")
    print("[INFO] 새로운 파일 생성 중...")

    out_good = filename.replace(".jsonl", f".score_ge_{threshold}.jsonl")
    out_bad  = filename.replace(".jsonl", f".score_lt_{threshold}.jsonl")

    with open(filename, "r", encoding="utf-8") as fin, \
         open(out_good, "w", encoding="utf-8") as fgood, \
         open(out_bad, "w", encoding="utf-8") as fbad:

        for idx, line in enumerate(fin):
            if idx in bad_idx:
                fbad.write(line)
            else:
                fgood.write(line)

    print(f"[INFO] 필터링 완료.")
    print(f" - 남긴 데이터   : {out_good}")
    print(f" - 제거된 데이터 : {out_bad}")

