# MPS 멀티잡 실행 런북 (이 레포 전용 · 실측 검증됨)

> 한 노드 8장에서 학습 **2개를 동시에** 돌릴 때 NVIDIA **MPS**로 속도 손실 없이 띄우는 절차.
> 2026-06-15 이 노드(8× B200)에서 실제 측정으로 검증함. 이론/원인은 [GPU_MULTIJOB_NOTES.md](GPU_MULTIJOB_NOTES.md) 참고.

---

## 0. 왜 MPS인가 (실측 요약)

같은 config(`test_traj` + `test_traj_mcls`) 2잡을 8장에서 동시에 돌린 결과:

| 조건 | 잡당 iter | 노드 총 처리량 | 결론 |
|---|---|---|---|
| 1잡 단독 | 8.2초 | 0.122 iter/s | 기준 |
| **둘 다 non-MPS** (그냥 겹침) | **30초** | 0.066 iter/s | ❌ 단독보다도 느림 |
| **한쪽만 MPS** | 32초 | ~0.062 iter/s | ❌ 효과 0 (제로섬) |
| **둘 다 MPS** ✅ | **9.0초** | **0.220 iter/s** | ✅ **3.3배 빠름, 단독 대비 1.79배** |

→ **둘 다 MPS면 잡당 30초 → 9초.** 두 잡 동시에 돌아도 거의 단독 속도(8.2초), 비용은 잡당 ~11%뿐.
이게 잘 되는 이유: 이 워크로드는 단독 GPU 사용률이 24~60%(launch-bound)라 SM이 비어서, MPS가 두 잡 커널을 합쳐 빈 SM을 채움.

---

## 1. 깨지면 안 되는 핵심 규칙 4가지

1. **두 잡 모두 MPS 클라이언트여야 함.** 한쪽만 켜면 효과 0 (실측 확인). GPU 입장에서 non-MPS 컨텍스트와 MPS 서버 컨텍스트가 여전히 시분할되기 때문.
2. **이미 떠 있는 잡은 MPS에 못 붙음.** CUDA 컨텍스트 생성 시점에 결정됨 → 데몬 나중에 켜도 소급 안 됨. **두 잡 다 새로 시작**해야 함.
3. **잡마다 `PORT` 다르게, `--work-dir` 다르게.** (rendezvous/로그 충돌 방지)
4. **메모리는 합산됨.** 잡당 ~65GB × 2 = 130GB < 183GB라 이 노드는 여유. (다른 노드면 확인)

---

## 2. 복붙 시퀀스 (happy path)

> ✅ **상시적용됨(2026-06-15~):** `tools/dist_train.sh`가 학습 시작 시 MPS 데몬을 **자동으로 띄움**.
> 따라서 보통은 **2-1 단계를 직접 안 해도 되고**, 그냥 두 잡을 다른 PORT로 띄우면 자동으로 MPS 클라이언트가 됨.
> MPS를 끄고 싶은 launch는 앞에 `USE_MPS=0` 붙이면 됨. 데몬 종료는 `bash stop_mps.sh`.

### 2-1. MPS 데몬 켜기 (노드당 딱 1번) — *이제 dist_train.sh가 자동으로 함, 수동은 선택*

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d            # 백그라운드 데몬으로 뜸 (sudo 불필요, 사용자 권한)

# 확인: 100.0 나오면 정상
echo get_default_active_thread_percentage | nvidia-cuda-mps-control
```

> ⚠️ `/tmp/nvidia-mps`는 MPS **기본 경로**라, 데몬이 떠 있으면 이후 시작되는 **모든** CUDA 프로세스가 자동으로 MPS에 붙는다. 같은 노드에서 남이 잡을 띄우면 그것도 붙을 수 있으니 공유 노드면 주의.

### 2-2. 두 잡 시작 (둘 다 같은 MPS env + 다른 PORT)

```bash
cd /NHNHOME/WORKSPACE/0526040009_A/hwanhee/Autonomous_Driving_26_ksh

# ── Job 1 (control: test_traj) ──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=22060 \
  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log \
  nohup bash tools/dist_train.sh ./projects/configs/baselines/test_traj.py 8 \
    --work-dir work_dirs/test_traj > /tmp/run_test_traj.log 2>&1 &

# ── Job 2 (mcls: test_traj_mcls) ──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=22061 \
  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log \
  nohup bash tools/dist_train.sh ./projects/configs/baselines/test_traj_mcls.py 8 \
    --work-dir work_dirs/test_traj_mcls > /tmp/run_test_traj_mcls.log 2>&1 &
```

포인트:
- **두 잡 다** `CUDA_MPS_PIPE_DIRECTORY` 세팅 (이게 MPS 클라이언트로 만드는 핵심).
- **두 잡 다** `CUDA_VISIBLE_DEVICES=0..7` (8장 전부 공유가 목적).
- `PORT`만 다르게(22060 / 22061), `--work-dir`도 다르게.
- `nohup ... &`로 터미널 닫혀도 유지. (또는 tmux/screen 권장)

### 2-3. MPS 붙었는지 확인 (둘 다 클라이언트인지)

```bash
# (a) 서버 PID 나오면 클라이언트가 붙은 것
echo get_server_list | nvidia-cuda-mps-control

# (b) python 16개(8+8) + nvidia-cuda-mps-server 가 보이면 둘 다 MPS ✓
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader \
  | awk -F',' '{print $2}' | sort | uniq -c
#   → "16 .../python" + "8 nvidia-cuda-mps-server" 이런 식이면 정상
```

> 만약 `nvidia-cuda-mps-server`가 안 보이고 python만 16개면 → 둘 다(또는 한쪽이) MPS에 **안 붙은** 것. env 빠졌는지 확인 후 재시작.

### 2-4. 속도 확인 (잘 되면 잡당 ~9초)

```bash
# 각 잡 최근 iter 시간 (cold-start 몇 개 지나고 봐야 함)
for d in work_dirs/test_traj work_dirs/test_traj_mcls; do
  f=$(ls -t $d/*.log.json | head -1)
  echo "== $d =="
  python -c "
import json,sys,statistics as st
xs=[json.loads(l)['time'] for l in open('$f') if '\"mode\": \"train\"' in l]
w=xs[-15:]
print(f'  최근{len(w)}개 평균 {st.mean(w):.1f}s (단독≈8.2s, 둘다MPS 목표≈9s)')"
done
```

### 2-5. 종료 / 정리 (⚠️ 순서 중요)

```bash
# 1) 학습 잡 종료 (port로 정확히, [b]racket으로 자기자신 매칭 방지)
pkill -9 -f "[t]rain.py.*work_dir.*test_traj"     # 필요시 PORT=2206[01] 패턴으로

# 2) MPS 데몬 종료 — 반드시 pipe dir 살아있을 때 quit!
echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nvidia-cuda-mps-control

# 3) (quit이 안 먹으면) 강제 종료
pkill -9 -f "[n]vidia-cuda-mps"

# 4) 마지막에 정리
rm -rf /tmp/nvidia-mps /tmp/nvidia-mps-log

# 확인: 메모리 ~4MiB, util 0이면 완전 클린
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## 3. 함정 / 트러블슈팅 (실제로 겪은 것)

| 증상 | 원인 | 해결 |
|---|---|---|
| MPS 켰는데 안 빨라짐 (한쪽만 32초) | **한쪽 잡만** MPS 클라이언트 | 두 잡 다 `CUDA_MPS_PIPE_DIRECTORY` 줘서 새로 시작 |
| 이미 돌던 잡이 MPS 효과 못 봄 | 컨텍스트 생성 후엔 MPS 가입 불가 | 그 잡을 **재시작**해야 함 |
| `echo quit`이 데몬에 안 닿음 | **pipe dir를 먼저 지워버림** | 정리 순서: quit → 그 다음 rm. 이미 꼬였으면 `pkill -9 -f "[n]vidia-cuda-mps"` |
| `pkill -f "..."`가 자기 셸까지 죽임 | 패턴이 pkill 명령줄 자신과 매칭 | 패턴 첫 글자를 대괄호로: `[t]rain.py`, `[n]vidia-cuda-mps` |
| iter 시간이 8초~50초로 들쭉날쭉 | non-MPS 시분할의 버스트(겹침/데이터로딩 위상차) | 10~18개 평균은 못 믿음. **40개+** 평균으로 볼 것. MPS 켜면 이 버스트 자체가 사라짐(std 12s→0.9s) |
| `cannot import name 'occ_pool_ext' ... circular import` | MPS와 무관. CUDA 확장 미빌드 | `cd projects/occ_plugin/ops/occ_pooling && python setup.py build_ext --inplace` |

---

## 4. 빠른 결정 가이드

- **두 실험(A/B)을 같이 돌리고 싶다** → ✅ **8+8 MPS** (이 런북). 둘 다 풀 8-GPU/배치 유지 → A/B 비교 깨끗 + 거의 단독 속도.
- **4+4로 나누면?** → 비추. `samples_per_gpu=1`이라 글로벌 배치가 8→4로 반토막 → 학습 레짐 바뀌어 A/B 오염. 잡당 속도도 8-GPU보다 느림.
- **단독 GPU util이 이미 90%+인 무거운 워크로드** → MPS 이득 작음(빈 SM이 없음). 그땐 분할이나 순차가 나을 수 있음. (이 레포 워크로드는 24~60%라 MPS가 크게 이득)

---

## 부록: 이 노드 실측 환경
- 8× B200 (183GB), 풀 NVLink, 72코어 2-NUMA, 단독 잡 GPU util 24~60%(launch-bound).
- 측정: 각 조건 40 iter 평균. 둘 다 MPS에서 std 0.9s로 매우 안정적 → MPS가 시분할을 진짜 동시실행으로 대체함을 확인.
