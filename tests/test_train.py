from llms.student.train import claim_output_dir


def _touch(out, *names):
    for n in names:
        p = out / n
        if n.endswith(".json"):
            p.write_text("{}")
        else:
            (p / "x").mkdir(parents=True)


def test_same_key_keeps_work_for_resume_and_reuse(tmp_path):
    assert not claim_output_dir(tmp_path, "key-a")  # first run: nothing to reuse
    _touch(tmp_path, "checkpoints", "adapter", "train_info.json")  # training wrote outputs, then the VM died
    assert claim_output_dir(tmp_path, "key-a")      # same key: resume / reuse
    assert (tmp_path / "checkpoints").exists() and (tmp_path / "train_info.json").exists()


def test_new_key_or_untagged_dir_starts_over(tmp_path):
    claim_output_dir(tmp_path, "key-a")
    _touch(tmp_path, "checkpoints", "adapter", "merged", "train_info.json")
    assert not claim_output_dir(tmp_path, "key-b")  # upstream changed
    assert not any((tmp_path / n).exists() for n in ("checkpoints", "adapter", "merged", "train_info.json"))
    _touch(tmp_path, "checkpoints")
    (tmp_path / "run_key.txt").unlink()             # --force drops the tag
    assert not claim_output_dir(tmp_path, "key-b") and not (tmp_path / "checkpoints").exists()
