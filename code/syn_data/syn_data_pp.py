import json

def preprocess_mijeong_and_null_to_none(src_path: str, dst_path: str):

    total_lines = 0
    sys_updated = 0
    assistant_parsed = 0
    assistant_parse_fail = 0
    who_changed = 0
    when_changed = 0

    with open(src_path, encoding="utf-8") as fin, \
         open(dst_path, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            total_lines += 1
            sample = json.loads(line)

            messages = sample.get("messages", [])
            if not messages:
                # messages가 없으면 그대로 저장
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            # 1) system 프롬프트 내 문구 수정: '미정' -> null
            #    (프롬프트 정의를 null 기준으로 바꾸기 위함)
            system_msg = messages[0].get("content", "")
            if "'미정'" in system_msg:
                # 단순 치환: '미정' → null
                # 필요하면 문맥에 맞게 더 정교하게 바꿔도 됨
                system_msg = system_msg.replace("'미정'", "null")
                messages[0]["content"] = system_msg
                sys_updated += 1

            # 2) assistant 출력(JSON 문자열) 내부 정규화
            assistant_msg = messages[-1].get("content", "")
            try:
                assistant_json = json.loads(assistant_msg)
            except json.JSONDecodeError:
                # JSON이 깨져 있으면 손대지 않고 통과
                assistant_parse_fail += 1
            else:
                assistant_parsed += 1
                tasks = assistant_json.get("tasks", [])

                for t in tasks:
                    # who 필드 처리
                    if "who" in t and t["who"] in ("미정", "null"):
                        t["who"] = None
                        who_changed += 1

                    # when 필드 처리
                    if "when" in t and t["when"] in ("미정", "null"):
                        t["when"] = None
                        when_changed += 1

                # 수정된 assistant_json을 다시 문자열로 넣기
                messages[-1]["content"] = json.dumps(
                    assistant_json,
                    ensure_ascii=False,
                    indent=2
                )

            # 수정된 sample을 한 줄로 다시 저장
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("=== 전처리 요약 ===")
    print(f"총 라인 수                 : {total_lines}")
    print(f"system 프롬프트 수정 개수   : {sys_updated}")
    print(f"assistant JSON 파싱 성공 개수: {assistant_parsed}")
    print(f"assistant JSON 파싱 실패 개수: {assistant_parse_fail}")
    print(f"who 필드 변경 개수          : {who_changed}")
    print(f"when 필드 변경 개수         : {when_changed}")


if __name__ == "__main__":
    src = "syn_data.jsonl"               # 원본 파일 경로
    dst = "syn_data_normalized_null.jsonl"  # 전처리 결과 파일 경로
    preprocess_mijeong_and_null_to_none(src, dst)
