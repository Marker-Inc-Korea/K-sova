# K-SOVA Archive

이 디렉터리는 현재 v2.0 이전의 데이터, 실험 결과, 실행 도구와 중간 설계를 삭제하지 않고
보존한다. 현행 프로젝트 설명과 재현 기준은 루트 [`README.md`](../README.md), 현행 데이터
상태는 [`benchmark/v2.0/CURRENT_MANIFEST.json`](../benchmark/v2.0/CURRENT_MANIFEST.json)을
따른다.

| 경로 | 보존 내용 |
|---|---|
| [`legacy_training/`](legacy_training/) | 기존 1~5차 학습 실험과 공용 파이프라인 |
| [`benchmark_v1/`](benchmark_v1/) | Bench v1.0 데이터·원천·결과·재현 도구와 당시 README |
| [`sovereign_alignment_v4/`](sovereign_alignment_v4/) | V4 설계·실험·보고서·도구 |
| [`v2_pilots/`](v2_pilots/) | 150문항, sharp-conflict, 1,000후보 설계 등 v2.0 중간 파일럿 |

아카이브는 이력과 감사용이며 현행 입력으로 사용하지 않는다. 각 버전은 쟁점 단위, 문항 수,
지식통제 포함 여부, 모델 구성과 점수 계약이 다르므로 숫자를 v2.0과 직접 비교하거나 합산하지
않는다. 재실험이 필요하면 원본을 수정하지 말고 별도의 새 실행 디렉터리를 만든다.
