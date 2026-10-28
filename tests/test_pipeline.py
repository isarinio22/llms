from llms.pipeline import STAGES, downstream


def test_downstream_is_transitive_and_ordered_registry():
    assert downstream("train_student") == {"learning_curve", "score", "evaluate", "cascade", "quantize", "serve_bench"}
    assert downstream("cascade") == set()
    assert "baseline" in downstream("data") and "teacher_label" in downstream("data")


def test_dependencies_come_before_dependents():
    seen: set[str] = set()
    for s in STAGES:
        assert set(s.deps) <= seen, f"{s.name} listed before its dependencies"
        seen.add(s.name)
