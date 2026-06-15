# GPU 멀티잡 운용 가이드 — 한 노드에서 학습 여러 개 동시에 돌리기

> 한 노드의 GPU N장에서 학습 잡을 2개 이상 동시에 띄울 때 생기는 **속도 저하 원인과 해결법** 정리.
> 다른 레포에서도 그대로 적용 가능. (이 레포에서 실측한 구체 수치는 맨 아래 [부록](#부록-이-환경에서-실측한-값) 참고.)

---

## TL;DR

- **증상**: 8 GPU 노드에서 학습 1개 → iter ~9s. 학습 2개 동시 → iter ~40s (2배가 아니라 **4배**).
- **원인**: 두 잡이 **같은 물리 GPU들을 똑같이 점유**(oversubscription). 각 GPU에 CUDA 컨텍스트 2개가 올라가 **시분할(time-slice)** + 컨텍스트 스위칭 오버헤드.
- **데이터/디스크/네트워크 문제 아님.** 로그상 데이터 로딩(`data_time`)은 전체의 ~6%뿐, 나머지 ~90%가 compute → 느려진 건 compute가 부푼 것.
- **"두 잡이 각각 8장 다 쓰기"는 환상.** GPU는 8장뿐이라 물리적으로 불가능하고, 겹치면 노드 총 처리량이 오히려 **절반**으로 떨어짐.
- **해결**:
  1. **GPU 분할** (예: 4+4) — 각 잡에 전용 GPU. 가장 단순·안전, 낭비 0. ✅ 기본 추천
  2. **MPS + 공유** — GPU 사용률에 여유가 있을 때 8+8로도 거의 안 느리게 가능 (측정 필요)
  3. **순차 실행** — 잡A 8장 → 끝나면 잡B 8장. 첫 결과를 빨리 받고 싶을 때

---

## 1. 원인: GPU Oversubscription

DDP 학습은 보통 `torchrun --nproc_per_node=K` + `CUDA_VISIBLE_DEVICES`로 띄운다.
디바이스 선택은 보통 다음과 같이 결정된다:

```python
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

→ `LOCAL_RANK`가 `CUDA_VISIBLE_DEVICES` 목록의 인덱스로 매핑된다.

**문제**: 두 잡을 **둘 다 `CUDA_VISIBLE_DEVICES=0..7` + `nproc_per_node=8`** 로 띄우면,
두 잡의 rank-k 프로세스가 **같은 물리 GPU k**에 올라간다 → GPU 1장당 프로세스 2개.

```
GPU0  ← 잡A rank0  +  잡B rank0   (2개가 한 GPU를 시분할)
GPU1  ← 잡A rank1  +  잡B rank1
...
```

- GPU `compute_mode=Default`이면 이 겹침을 **막지 않는다** (조용히 허용 → 그냥 느려짐).
- MPS가 없으면 GPU가 두 컨텍스트를 **번갈아** 실행하고, 전환할 때마다 SM/캐시 상태를 비우고 다시 채운다.

---

## 2. 왜 2배가 아니라 3~4배인가 (요인들이 곱셈)

1. **GPU 직렬 시분할 (×~2)**: GPU가 잡A 커널 → 잡B 커널 순차 실행.
2. **컨텍스트 스위치 오버헤드 (×~1.3~1.6)**: 단일 잡일 때 이미 GPU 사용률이 낮으면(=launch/latency-bound),
   스텝당 **고정 오버헤드**가 지배적. 컨텍스트 2개면 이 고정비를 두 번 내고 + 전환 비용까지 → super-linear.
3. **CPU/NUMA 가중**: 각 rank의 Python 메인스레드는 GIL로 코어 1개를 점유. 잡 2개면 메인스레드가 2배로 늘고,
   **핀 고정 없이** 멀티 소켓(NUMA)을 떠돌면 cross-NUMA 메모리 접근 + 스케줄러 마이그레이션으로 추가 지연.

대략 `2 × 1.4 × … ≈ 3~4배`. GPU 공유가 주범, CPU/NUMA가 곱셈 가중.

> **핵심 직관**: 단일 잡이 GPU를 100% 못 쓰고 있을수록(launch-bound) oversubscription 페널티가 더 커진다.

---

## 3. 진단 — 범인 확정하기 (몇 분)

### (a) "데이터냐 compute냐" 먼저 가르기 — 가장 중요
mmcv/mmdet 로그(`*.log.json`)에서 정상 상태 iter(첫 iter는 cold-start라 제외)의 `time`과 `data_time`을 본다:

```bash
python -c "
import json,sys
xs=[json.loads(l) for l in open(sys.argv[1]) if '\"mode\": \"train\"' in l]
xs=xs[2:]  # cold-start 제외
t=sum(x['time'] for x in xs)/len(xs)
d=sum(x.get('data_time',0) for x in xs)/len(xs)
print(f'time={t:.2f}s data_time={d:.2f}s compute={t-d:.2f}s ({100*d/t:.0f}% data)')
" work_dirs/<run>/<stamp>.log.json
```

- `data_time`이 작고(예: <10%) `time`만 커짐 → **compute 병목 = GPU 공유/CPU**. (데이터 무죄)
- `data_time` 자체가 큼 → 데이터 파이프라인/디스크 의심.

> ⚠️ **함정**: 첫 1~2 iter는 numba JIT 컴파일 + 콜드 캐시로 수십 초가 정상. 이걸 평균에 넣으면
> "data_time 폭증"처럼 착시가 생긴다. 반드시 cold-start iter를 빼고 본다.

### (b) 프로세스 → 물리 GPU 매핑 (두 잡 다 떠 있을 때)
```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
nvidia-smi --query-gpu=index,uuid --format=csv
# uuid로 조인 → 같은 gpu_uuid에 서로 다른 PID 2개면 = oversubscription 확정
```
또는 `nvidia-smi` 하단 Processes에 python이 GPU 수의 2배면 겹친 것.

### (c) GPU 사용률 거동
```bash
nvidia-smi dmon -s u -d 1
```
- 사용률이 높게 붙는데 처리량은 반토막 + 톱니 → 커널 큐잉(시분할). **GPU 공유**.
- 사용률이 0 쪽으로 무너지고 CPU가 꽉 참 → **데이터/호스트 starvation**.

### (d) CPU/NUMA 압박
```bash
cat /proc/loadavg          # 잡 2개에서 코어 수 근처/이상이면 CPU 포화
mpstat -P ALL 2 3          # 코어별 포화
numastat                   # other_node / numa_foreign 증가 → cross-NUMA thrash
```

### (e) 결정적 A/B 테스트
GPU를 쪼개서(분할) 다시 띄워 iter 시간이 단일 잡 수준으로 돌아오면 → **GPU 공유가 범인 확정.**

---

## 4. 해결책

### 옵션 1 — GPU 분할 (기본 추천) ✅
각 잡에 **전용 GPU**를 주고, **`nproc_per_node`를 보이는 GPU 수와 일치**시킨다.
포트도 잡마다 다르게(rendezvous 충돌 방지).

```bash
# Job A
CUDA_VISIBLE_DEVICES=0,1,2,3  MASTER_PORT=29501  torchrun --nproc_per_node=4 ... 
# Job B
CUDA_VISIBLE_DEVICES=4,5,6,7  MASTER_PORT=29502  torchrun --nproc_per_node=4 ...
```

> ⚠️ `nproc_per_node`를 보이는 GPU 수보다 크게 두면(예: GPU 4장에 8) **한 잡 안에서 또 oversubscription**이 생긴다.

**"낭비 아니냐?" → 아니다.** 8장을 4+4로 나눠도 8장 전부 일하고 있다. 노드 전체 처리량은:

| 방식 | iter 시간 | 노드 총 처리량 | 비고 |
|---|---|---|---|
| 8+8 겹치기 | ~40s | 16 samples / 40s ≈ **0.4/s** | 절반 이상 오버헤드로 날림 |
| **4+4 분할** | ~9~12s | 8 samples / ~10s ≈ **0.8/s** | 8장 100% 사용, **2배 일함** |

분할이 겹치기보다 **총 처리량 2배**. 두 실험을 다 끝내는 시간도 더 짧다.

### 옵션 2 — MPS 켜고 공유 (8+8로도 빠르게)
**단일 잡 GPU 사용률이 낮을 때(여유가 많을 때)** 효과적. MPS는 여러 프로세스의 커널을 **하나의 창구로 합쳐 동시 실행**시켜 빈 SM을 채운다.

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
nvidia-cuda-mps-control -d                 # 데몬 시작 (사용자 권한으로 가능)
# 이제 두 잡 모두 CUDA_VISIBLE_DEVICES=0..7 로 평소처럼 실행 (MASTER_PORT만 다르게)
echo quit | nvidia-cuda-mps-control        # 끌 때
```

- 메모리는 **합산**된다 (잡당 메모리 × 2 < GPU 메모리여야 함).
- 효과는 워크로드 의존 → **켜고 iter 시간을 분할 방식과 비교 측정**할 것.
- 병목이 GPU가 아니라 CPU(메인스레드)면 MPS만으론 한계.
- 격리가 약함: 한 잡이 죽으면 MPS 통해 다른 잡에 영향 줄 수 있음.

### 옵션 3 — 순차 실행
잡A를 8장으로 끝내고 → 잡B를 8장으로. 총 시간은 분할과 비슷하지만 **첫 결과가 절반 시점에** 나온다.
한 결과를 빨리 봐야 할 때 유리.

### 보너스 — CPU/NUMA 핀 고정 (어느 옵션이든 같이)
멀티 소켓 노드면 각 잡을 자기 GPU와 같은 NUMA 노드의 코어/메모리에 고정한다.

```bash
# 예: GPU 0-3이 NUMA node0, GPU 4-7이 node1인 경우
numactl --cpunodebind=0 --membind=0  <Job A launch>   # GPU 0-3
numactl --cpunodebind=1 --membind=1  <Job B launch>   # GPU 4-7
```
GPU↔NUMA 매핑은 `nvidia-smi topo -m`의 "NUMA Affinity" 열로 확인.

---

## 5. MPS란 (NVIDIA Multi-Process Service)

GPU 한 장을 여러 프로세스가 **동시에** 효율적으로 나눠 쓰게 하는 NVIDIA 기능.

- **기본 모드**: GPU가 두 프로세스를 **번갈아**(time-slice) 처리 → 전환 비용 발생.
- **MPS**: 두 프로세스의 작업을 한 창구로 합쳐 **진짜 동시** 실행 → 비는 SM을 메움.
- **비유**: 36차선 도로. 기본 모드는 잡A 차가 다 지나갈 때까지 잡B 대기(차선 절반 비어도). MPS는 빈 차선으로 잡B가 같이 달림.
- **언제 이득**: 단일 잡 GPU 사용률이 낮을수록(빈 차선 많을수록) 크다.

확인:
```bash
nvidia-smi --query-gpu=index,compute_mode --format=csv   # Default면 겹침 허용
echo $CUDA_MPS_PIPE_DIRECTORY; pgrep -a nvidia-cuda-mps   # MPS 데몬 떠있나
```

---

## 6. 빠른 체크리스트

새 노드/레포에서 멀티잡 돌리기 전:

- [ ] `nvidia-smi topo -m` — GPU 수, NVLink/PCIe, GPU↔NUMA 매핑 확인
- [ ] `nproc` / `lscpu` — 코어 수, 소켓(NUMA) 수
- [ ] 단일 잡 로그로 `time` vs `data_time` 분해 → 병목이 compute인지 data인지
- [ ] 단일 잡 `nvidia-smi dmon -s u` — GPU 사용률 (낮으면 MPS 후보)
- [ ] 두 잡 띄울 때: **GPU 분할 + 잡마다 다른 MASTER_PORT** 가 기본
- [ ] `nproc_per_node == 그 잡에 보이는 GPU 수` 확인
- [ ] 멀티 소켓이면 NUMA 핀 고정
- [ ] 8장 공유를 원하면 MPS 켜고 분할 방식과 **실측 비교**

---

## 7. 자주 하는 오해

| 오해 | 사실 |
|---|---|
| "두 잡 각각 8장 쓰면 빠르다" | 물리적으로 불가능. 겹치면 시분할 + 오버헤드로 **총 처리량 절반** |
| "4+4는 GPU 낭비" | 8장 전부 사용 중. 겹치기보다 **총 처리량 2배** |
| "느린 건 데이터 로딩/디스크 탓" | 로그상 data_time이 작으면 무죄. compute(=GPU 공유)가 범인 |
| "첫 iter가 수십 초 = 데이터 문제" | numba JIT + 콜드 캐시 워밍업. 정상. 평균에서 제외하고 봐야 함 |

---

## 부록: 이 환경에서 실측한 값

> 아래는 **이 노드 한정** 수치. 다른 레포/노드에선 위의 원리/명령으로 다시 측정할 것.

- **하드웨어**: 8× NVIDIA B200 (183GB), **풀 NVLink 메시(NV18)** → 인터커넥트 경합 없음.
  CPU 2× AMD EPYC 9365 36c = **72코어, SMT 없음, 2 NUMA**(node0=cpu0-35/GPU0-3, node1=cpu36-71/GPU4-7).
  RAM 2.2TB, `/dev/shm` 600G — 전부 여유.
- **데이터**: Lustre(IB) 공유 FS. 8-way 동시 대용량 읽기 거의 선형 스케일 → **2-job 슬로다운의 원인 아님**.
- **단일 잡 정상 상태**: `time≈8.9s`, `data_time≈0.5s`, `compute≈8.1s` (data ~6%). GPU 사용률 24~60% (= launch-bound, MPS 여지 큼).
- **결론**: 9s→40s는 100% compute 부풀음 = 두 잡이 8장을 겹쳐 점유한 oversubscription.
- **검증됨**: 사용자가 GPU 4+4로 분할하니 40s까지 안 감 → GPU 공유가 원인으로 확정.
- **참고**: 이 레포 런처는 `train_total.sh`가 `CUDA_VISIBLE_DEVICES=0..7` + `PORT=22036`을 하드코딩.
  두 번째 잡은 PORT를 바꿔야 뜨고, GPU도 분할하려면 CUDA_VISIBLE_DEVICES를 잡마다 다르게 줘야 함.
