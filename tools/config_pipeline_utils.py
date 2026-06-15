def _disable_occ_dt_in_pipeline(pipeline):
    if not isinstance(pipeline, (list, tuple)):
        return

    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "LoadOccupancy":
            step["load_occ_dt"] = False
        elif step.get("type") == "Collect3D" and "keys" in step:
            step["keys"] = [k for k in step["keys"] if k != "occ_dt"]


def _walk_dataset_cfg(data_cfg):
    if isinstance(data_cfg, (list, tuple)):
        for item in data_cfg:
            _walk_dataset_cfg(item)
        return
    if not isinstance(data_cfg, dict):
        return

    if "pipeline" in data_cfg:
        _disable_occ_dt_in_pipeline(data_cfg["pipeline"])
    if "dataset" in data_cfg:
        _walk_dataset_cfg(data_cfg["dataset"])
    if "datasets" in data_cfg:
        _walk_dataset_cfg(data_cfg["datasets"])


def sync_occ_dt_loading_with_model_cfg(cfg):
    model_cfg = cfg.get("model", {}).get("model_cfg", {})
    if bool(model_cfg.get("use_query_dt_loss", False)):
        return

    data_cfg = cfg.get("data", {})
    for split in ("train", "val", "test"):
        if split in data_cfg:
            _walk_dataset_cfg(data_cfg[split])
