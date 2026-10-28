from llms.student.train import reset_stale_checkpoints


def test_checkpoints_survive_same_key_and_are_reset_on_new_key(tmp_path):
    assert not reset_stale_checkpoints(tmp_path, "key-a")  # first run: nothing to reset
    ckpt = tmp_path / "checkpoints" / "checkpoint-10"
    ckpt.mkdir(parents=True)                               # training writes a checkpoint, then Colab disconnects
    assert not reset_stale_checkpoints(tmp_path, "key-a")  # same key: resume
    assert ckpt.exists()
    assert reset_stale_checkpoints(tmp_path, "key-b")      # upstream changed: start over
    assert not ckpt.exists() and (tmp_path / "run_key.txt").read_text() == "key-b"


def test_untagged_checkpoints_are_stale(tmp_path):
    (tmp_path / "checkpoints" / "checkpoint-5").mkdir(parents=True)
    assert reset_stale_checkpoints(tmp_path, "key-a")  # unknown provenance (or --force dropped the tag)
    assert not (tmp_path / "checkpoints").exists()
