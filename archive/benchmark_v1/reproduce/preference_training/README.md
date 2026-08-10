# 선호학습 후속 실험 재현

이 디렉터리는 2026년 7월 31일 기준 K-SOVA 후속 선호학습 실험에서 사용한 재사용 가능한 Python 스크립트만 보관한다. 원시 모델 응답, 판정 응답, 체크포인트, 서버 로그와 클러스터 절대경로가 포함된 일회성 Slurm 스크립트는 공개하지 않는다.

## 스크립트

| 파일 | 역할 |
|---|---|
| `build_v26_source_extension.py` | 공식 원천 추가 수집과 평가 원천 URL 중복 검사 |
| `filter_conflict_objects_by_sources.py` | 허용된 훈련 원천만 사용하는 갈등 객체 선별 |
| `split_conflict_objects.py` | 원천 연결 성분 단위 훈련·검증 분리 |
| `expand_v26_author_requests.py` | 영역·변형 축 균형을 유지한 정확 수량 생성 요청 확장 |
| `build_v26_dpo_dataset.py` | 이중 검수 문항을 선호쌍으로 변환하고 평가셋 유사도 차단 |
| `train_qwen_dpo.py` | Qwen3.5 계열 LoRA DPO 학습과 선호 정확도 평가 |

문항 생성·검수의 핵심 계약은 `../../tools/conflict_object_pipeline.py`와 `../../tools/core_pipeline.py`에 있다. 모델 서빙과 배치 호출은 상위 [평가 재현 안내](../README.md)의 `run_vllm_stage.sbatch` 및 `tools/chat_batch.py` 패턴을 사용한다. 모든 GPU 작업은 Slurm 할당 안에서 실행해야 한다.

## 실행 순서

1. 공식 원천을 확장하고 공개 평가 원천과 URL 중복이 0인지 확인한다.
2. 갈등 객체를 독립 모델로 감사한 뒤 허용된 원천만 남긴다.
3. 원천 연결 성분 단위로 훈련·검증 객체를 분리한다.
4. Gemma 저자 요청을 목표 수량만큼 확장하고 `core_pipeline.py parse-author`로 정적 검수한다.
5. 서로 다른 모델로 구성타당성과 근거성을 독립 검수한다.
6. 두 검수를 모두 통과한 문항만 중복 제거하고 선호쌍으로 변환한다.
7. DPO 학습 뒤 고정 홀드아웃에서 base와 동일 생성 조건으로 비교한다.

현재 확장 사이클은 5단계 근거 검수가 미완료이므로 6단계 이후를 실행하지 않았다. 수량과 실패 원인은 [진행 보고](../../results/preference_training_progress_20260731.md)를 따른다.
