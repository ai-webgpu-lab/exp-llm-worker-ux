# Results

## 1. 실험 요약
- 저장소:
- 커밋 해시:
- 실험 일시:
- 담당자:
- 실험 유형: `graphics | blackhole | ml | llm | audio | multimodal | agent | benchmark | integration | infra | docs`
- 상태: `success | partial | failed`

## 2. 질문
- 이 실험이 답하려는 핵심 질문 1
- 핵심 질문 2
- 핵심 질문 3

## 3. 실행 환경
### 브라우저
- 이름:
- 버전:

### 운영체제
- OS:
- 버전:

### 디바이스
- 장치명:
- device class: `desktop-high | desktop-mid | laptop | mobile-high | mobile-mid | unknown`
- CPU:
- 메모리:
- 전원 상태: `plugged | battery | unknown`

### GPU / 실행 모드
- adapter:
- backend: `webgpu | wasm | webgl | webnn | mixed`
- fallback triggered: `true | false`
- worker mode: `main | worker | shared | unknown`
- cache state: `cold | warm | mixed | unknown`
- required features:
- limits snapshot:

## 4. 워크로드 정의
- 시나리오 이름:
- 입력 프로필:
- 데이터 크기:
- dataset:
- model_id 또는 renderer:
- 양자화/정밀도:
- resolution:
- context_tokens:
- output_tokens:

## 5. 측정 지표
### 공통
- time_to_interactive_ms:
- init_ms:
- success_rate:
- peak_memory_note:
- error_type:

### Graphics / Blackhole
- avg_fps:
- p95_frametime_ms:
- scene_load_ms:
- resolution_scale:
- ray_steps:
- taa_enabled:
- visual_artifact_note:

### Embeddings
- docs_per_sec:
- queries_per_sec:
- p50_ms:
- p95_ms:
- recall_at_10:
- index_build_ms:

### RAG
- ingest_ms_per_page:
- chunk_count:
- embed_total_ms:
- retrieve_ms:
- rerank_ms:
- answer_ttft_ms:
- answer_total_ms:
- citation_hit_rate:

### LLM
- ttft_ms:
- prefill_tok_per_sec:
- decode_tok_per_sec:
- turn_latency_ms:

### STT
- audio_sec_per_sec:
- first_partial_ms:
- final_latency_ms:
- wer:
- cer:

### Voice
- roundtrip_ms:
- interrupt_recovery_ms:
- handsfree_success_rate:

### VLM
- image_preprocess_ms:
- image_to_first_token_ms:
- answer_total_ms:
- accuracy_task_score:

### Diffusion
- sec_per_image:
- steps_per_sec:
- resolution_success_rate:
- oom_or_fail_rate:

### Agent
- task_success_rate:
- avg_step_latency_ms:
- tool_call_success_rate:
- user_intervention_count:

## 6. 결과 표
| Run | Scenario | Backend | Cache | Mean | P95 | Notes |
|---|---|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |  |

## 7. 관찰
- 잘 된 점
- 예상보다 느린 구간
- 브라우저별 차이
- fallback 발생 조건
- 메모리/열/안정성 이슈

## 8. 결론
- 이번 실험으로 확인한 사실
- 아직 확정 못 한 부분
- 다음 실험에서 바꿔야 할 점

## 9. 첨부
- 스크린샷:
- 로그 파일:
- raw json:
- 배포 URL:
- 관련 이슈/PR:
