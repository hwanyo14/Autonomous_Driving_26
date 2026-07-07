# dbg_probe — 체크포인트 오프라인 진단/시각화 스크립트 모음 (2026-07-02~03)

학습 없이 체크포인트를 소수 train 샘플에 forward시켜 내부 텐서를 캡처·분석하는 일회성
프로브들. GPU ~3GB, 샘플당 수십 초. **공통 주의**:

- 파일 상단의 `REPO`/`CFG`/`CKPT`(및 `IDXS`, `N_BIG` 등)가 하드코딩 — 재사용 시 수정할 것.
- vis 게이트는 스크립트 안에서 끔(`debug_*_vis_every=0`) — 안 끄면 work_dirs/vis에 stray PNG 생김.
- 실행: `CUDA_VISIBLE_DEVICES=<빈GPU> ~/anaconda3/envs/eof/bin/python tools/dbg_probe/<스크립트>.py`
  (실행 전 `nvidia-smi`로 여유 확인 — MPS 공유 서버).
- 출력(npz/json/png)은 스크립트 내 `OUT` 경로로 — 원래 세션 scratchpad였으니 재사용 시
  `work_dirs/<run>/dbg_analysis/` 등으로 바꿔줄 것.

| 스크립트 | 하는 일 | 산출물 |
|---|---|---|
| `dbg_offset_spread.py` | matched query별 offset-spread/mixture 텐서 캡처 (ep1 vs ep10 등 다중 ckpt 비교) | cap_*.npz, spread_stats.json |
| `plot_spread_results.py` | 위 캡처로 크기-bin 통계 + 최대 instance BEV 플롯 | spread_stats.png, bev_biggest.png |
| `cap_we_bev.py` | 특정 샘플 mixture 캡처 + **두 run BEV 나란히 비교 플롯** (flag-pole 그림이 이것) | cap_*.npz, bev_flagpole_vs_ghost.png |
| `probe_we_spread.py` | visible-gate spread vs GT size 상관 + 가시 gaussian 수/반경 (loss 우회 판정) | we_spread_ep10.json |
| `probe_feat_size.py` | query feature에서 GT 크기가 선형 디코딩되는지 (ridge, group CV, 셔플 컨트롤) | feat_probe_ep10.npz |
| `probe_bev_size.py` | LSS BEV feature 고정창/instance-pool에서 크기 디코딩 여부 | bev_probe_ep10.npz |
| `probe_attn_geometry.py` | attn 폭(σ) vs 마스크 폭, coverage/inside-mass, σ×depth 크기 상관 | attn_geom_ep10.json |

주요 실측 결과와 해석은 NOTES.md 2026-07-02~03 항목 참조.
