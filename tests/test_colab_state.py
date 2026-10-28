from llms.colab import pack_state, restore_state

FILES = [
    "results/full/teacher.json",
    "figures/full/learning_curve.png",
    "logs/run.log",
    "artifacts/_shared/teacher_cache/cache_abc.jsonl",
    "artifacts/full/data/splits.parquet",
    "artifacts/full/data/embeddings.npy",
    "artifacts/full/data/_DONE.json",
    "artifacts/full/teacher_label/_DONE.json",
    "artifacts/full/train_student/_DONE.json",
    "artifacts/full/train_student/main/adapter/adapter_model.safetensors",
    "artifacts/full/train_student/main/train_info.json",
    "artifacts/full/train_student/main/run_key.txt",
    "artifacts/full/train_student/main/merged/model.safetensors",
    "artifacts/full/train_student/main/checkpoints/checkpoint-10/x.bin",
    "artifacts/full/learning_curve/curve_500/adapter/adapter_model.safetensors",
    "artifacts/full/learning_curve/curve_500/train_info.json",
    "artifacts/smoke/score/test.parquet",
]


def test_bundle_keeps_small_state_and_reopens_training_without_merged_model(tmp_path):
    vm, mac = tmp_path / "vm", tmp_path / "mac"
    for f in FILES:
        (vm / f).parent.mkdir(parents=True, exist_ok=True)
        (vm / f).write_text(f)
    n = pack_state(vm, "full", tmp_path / "state.tar.gz")
    restore_state(tmp_path / "state.tar.gz", mac, "full")
    got = {str(p.relative_to(mac)) for p in mac.rglob("*") if p.is_file()}
    assert {"results/full/teacher.json", "logs/run.log", "artifacts/_shared/teacher_cache/cache_abc.jsonl",
            "artifacts/full/teacher_label/_DONE.json", "artifacts/full/learning_curve/curve_500/train_info.json",
            "artifacts/full/train_student/main/adapter/adapter_model.safetensors"} <= got
    for excluded in ("embeddings.npy", "merged", "checkpoints", "curve_500/adapter", "artifacts/smoke",
                     "artifacts/full/data/_DONE.json"):
        assert not any(excluded in g for g in got), excluded
    # The merged model is never shipped, so train_student reopens to rebuild it from the adapter.
    assert "artifacts/full/train_student/_DONE.json" not in got
    assert n == len(got) + 1
