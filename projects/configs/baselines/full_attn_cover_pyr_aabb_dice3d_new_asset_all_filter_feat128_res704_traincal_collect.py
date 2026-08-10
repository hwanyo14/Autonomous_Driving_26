"""res704 traincal 정규화 통계 수집용 config.

`_feat128_res704` 를 그대로 상속하고 **`data.test.ann_file` 만 train 으로 바꾼다**.
`pipeline`(test_pipeline)·`test_mode` 는 손대지 않는다 — EVAL_METHOD.md 2.1 이
"train annotation 을 쓰되 학습용 random augmentation 이 아니라 validation 과 같은
deterministic test pipeline 으로 inference" 를 요구하기 때문이다.

사용:
    EOCF_EVAL_TRAINCAL_COLLECT=<out.json> bash tools/dist_test.sh <이 config> <ckpt> 1
    또는 CFG_NAME=..._feat128_res704 EP=18 ./collect_traincal_stats.sh

⚠ 이 config 로는 **metric 을 보지 말 것**. train split 이라 val 지표와 비교 불가이고,
  목적은 오직 `depth_std` / `past_speed` 의 log1p population mean/std 수집이다.
⚠ checkpoint 종속이다. res704 ep18 통계를 다른 에폭이나 다른 config 에 재사용하면
  z-score 가 통째로 틀어지는데 **에러가 나지 않는다**(NOTES 2026-08-03).
  실측: dim96 ep18 vs feat128 ep19 에서 speed std 가 −9% 어긋났다.
"""

_base_ = ['./full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128_res704.py']

# 상속 config 와 동일한 값 (여기서 다시 정의해야 _delete_ 없이 test 만 덮어쓸 수 있다)
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"

data = dict(
    test=dict(
        ann_file=train_ann_file,
        # train split 전체를 대상으로 둔다. 실제 수집량은 EOCF_EVAL_MAX_SAMPLES 로 조절하고,
        # 중간 스냅샷이 계속 저장되므로 4,000 지점에서 꺼내 쓰고 그대로 더 돌려도 된다.
        test_capacity=23930,
    ),
)
