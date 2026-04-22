# exp-llm-worker-ux

`worker UX 실험`를 단일 질문으로 분리해 검증하는 experiment 저장소입니다. 하나의 가설과 baseline에 집중하고, 결과를 재현 가능한 형식으로 남겨 이후 benchmark/app 저장소의 입력으로 사용합니다.

## 저장소 역할
- 한 저장소가 한 질문만 답하도록 범위를 좁혀 baseline을 빠르게 검증합니다.
- 구현 자체보다 가설 검증과 결과 기록을 우선합니다.
- 이후 benchmark와 app 저장소에서 재사용할 입력, 구현, 한계 사항을 명확히 남깁니다.

## 핵심 질문
- worker UX 실험를 브라우저/WebGPU 환경에서 재현 가능한 baseline으로 만들 수 있는가
- 현재 트랙에서 가장 먼저 확인해야 할 병목, 제약, fallback 조건은 무엇인가
- 이후 비교 또는 통합 단계로 넘기기 전에 어떤 최소 증거가 필요한가

## 포함 범위
- 단일 baseline 또는 소수의 명확한 비교 축
- 실행 환경, capability, raw 결과 기록
- `RESULTS.md`와 스크린샷/로그를 포함한 재현 가능한 검증

## 비범위
- 한 저장소 안에서 여러 질문을 동시에 푸는 것
- 제품 수준의 UX 완성도
- 재현 경로 없이 주장만 남는 결과 요약

## 기본 구조
- `src/` - 구현 코드 또는 baseline 프로토타입
- `public/` - GitHub Pages baseline probe 또는 실제 정적 데모 산출물
- `reports/raw/` - 원시 측정 결과 JSON/CSV/로그
- `reports/screenshots/` - 시각 결과 스크린샷
- `reports/logs/` - 실행 로그와 디버깅 산출물
- `schemas/ai-webgpu-lab-result.schema.json` - 공통 결과 스키마
- `RESULTS.md` - 핵심 결과 요약과 해석

## 메타데이터
- Track: LLM
- Kind: experiment
- Priority: P1

## 현재 상태
- Repository scaffold initialized
- Shared result schema copied to `schemas/ai-webgpu-lab-result.schema.json`
- Shared reporting template copied to `RESULTS.md`
- GitHub Pages baseline probe copied to `public/index.html`
- Browser probe logic copied to `public/app.js`
- GitHub Pages workflow copied to `.github/workflows/deploy-pages.yml`

## GitHub Pages 운영 메모
- Pages URL: https://ai-webgpu-lab.github.io/exp-llm-worker-ux/
- 기본 bootstrap workflow는 `public/` baseline probe 정적 artifact를 배포합니다.
- 실제 빌드가 필요한 저장소는 install/build 단계와 artifact 경로를 저장소 사양에 맞게 교체해야 합니다.

## 측정 및 검증 포인트
- first-token or first-result latency, steady-state throughput
- model load time, cache behavior, worker/off-main-thread 영향
- 품질, 오류율, fallback 경로, 사용자 체감 응답성

## 산출물
- 첫 runnable baseline 또는 최소 비교 구현
- raw 결과, 로그, 스크린샷
- `RESULTS.md` 기반의 요약과 해석

## 작업 및 갱신 절차
- `src/` 아래에 첫 runnable baseline 또는 비교 harness를 구현합니다.
- 실제 사용 스택이 정해지면 이 README에 install/dev/build/test 명령을 추가합니다.
- 측정 결과는 `reports/raw/`와 `RESULTS.md`에 함께 반영합니다.
- 브라우저, OS, 디바이스, cache, worker 여부 등 재현 조건을 결과와 같이 기록합니다.
- Pages를 유지하는 경우 baseline probe 또는 workflow를 실제 저장소 동작에 맞게 교체합니다.

## 완료 기준
- 핵심 가설에 대한 예/아니오 또는 조건부 결론을 낼 수 있습니다.
- raw 결과와 요약 문서가 함께 존재합니다.
- 다음 단계가 benchmark인지 app인지 README에서 판단할 수 있습니다.

## 관련 저장소
- `tpl-webgpu-vanilla` 또는 `tpl-webgpu-react` - baseline 출발점
- `shared-webgpu-capability` - capability/fallback 수집
- `shared-bench-schema`, `docs-lab-roadmap`

## 라이선스
MIT
