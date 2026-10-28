"""Stage 8: render every result table from results/<run>/*.json into Markdown.

Nothing here computes a new number: it only formats saved results, so every figure in the README is
traceable to a file. The full run's report replaces the block between the RESULTS markers in README.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from llms.config import Config
from llms.utils import log
from llms.workspace import Workspace, read_json

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NAMES = {"teacher": "Teacher", "student": "Student (LoRA)", "baseline_teacher_labels": "Embeddings + LR (teacher labels)",
         "baseline_gold_labels": "Embeddings + LR (gold labels, reference)"}


def _bad(x: Any) -> bool:
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def pct(x: Any, d: int = 1) -> str:
    return "n/a" if _bad(x) else f"{100 * x:.{d}f}%"


def num(x: Any, d: int = 3) -> str:
    return "n/a" if _bad(x) else f"{x:,.{d}f}"


def signed_pp(x: Any, d: int = 2) -> str:
    return "n/a" if _bad(x) else f"{100 * x:+.{d}f} pp"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fig(ws: Workspace, name: str, alt: str) -> str:
    path = ws.figures / name
    return f"![{alt}]({path.relative_to(ws.base).as_posix()})" if path.exists() else ""


def load_all(ws: Workspace) -> dict[str, dict]:
    names = ("data", "teacher", "baseline", "train_student", "learning_curve", "score", "evaluate", "cascade",
             "quantize", "serving", "env")
    return {n: read_json(ws.results / f"{n}.json") for n in names if (ws.results / f"{n}.json").exists()}


# Sections ---------------------------------------------------------------------------
def headline(R: dict, cfg: Config) -> str:
    if "evaluate" not in R:
        return ""
    m = R["evaluate"]["models"]
    model_ids = {"teacher": cfg.teacher.model, "student": cfg.student.model,
                 "baseline_teacher_labels": cfg.data.embedding_model, "baseline_gold_labels": cfg.data.embedding_model}
    rows = [[NAMES[k], f"`{model_ids[k]}`", pct(v["test"]["accuracy"]), num(v["test"]["macro_f1"])] for k, v in m.items()]
    if "cascade" in R:
        c = R["cascade"]["test"]
        rows.append(["Cascade (student, teacher fallback)", f"{pct(c['share_sent_to_teacher'])} to teacher",
                     pct(c["cascade_accuracy"]), "n/a"])
    return "### Headline (test set, n = {:,})\n\n".format(R["evaluate"]["n_test"]) + table(
        ["System", "Model", "Accuracy", "Macro-F1"], rows)


def data_section(R: dict) -> str:
    if "data" not in R:
        return ""
    d = R["data"]
    rows = [[s, d["split_sizes_before_dedup"][s], d["split_sizes_after_dedup"][s],
             sum(d["removed_by_split_and_reason"].get(s, {}).values())] for s in d["split_sizes_before_dedup"]]
    sens = ", ".join(f"cos ≥ {t}: {n}" for t, n in d["items_above_cosine_threshold"].items())
    return ("### Data and splits\n\n"
            f"{d['n_used']:,} queries, stratified split (seed {d['seed']}) before any teacher call. "
            f"Near-duplicates across splits (normalized text or cosine ≥ {d['dedup_threshold']} with "
            f"`{d['embedding_model']}`) removed: **{d['removed_total']}**. Sensitivity: {sens}.\n\n"
            + table(["Split", "Before dedup", "After dedup", "Removed"], rows))


def teacher_section(R: dict) -> str:
    if "teacher" not in R:
        return ""
    t = R["teacher"]
    tp, ps = t["train_pool"], t["per_split"]
    rows = [[s, pct(ps[s]["accuracy"]), num(ps[s]["macro_f1"]), ps[s]["no_valid_output"],
             " / ".join(f"{k}: {v}" for k, v in ps[s]["agree_distribution"].items())] for s in ("train_pool", "dev", "test") if s in ps]
    return ("### Teacher labeling\n\n"
            f"`{t['teacher']}` via {t['backend']}, k = {t['k']} samples at T = {t['temperature']}, "
            f"keep if ≥ {t['min_agree']} agree. First-round invalid outputs: {pct(t['first_round_invalid_rate'])}; "
            f"after retries: {pct(t['final_invalid_slot_rate'])}. Teacher calls this run: {t['calls']['samples']:,} samples "
            f"({t['calls']['cached_items_at_start']:,} items were already cached).\n\n"
            + table(["Split", "Accuracy vs gold", "Macro-F1", "No valid output", "Agreement (k samples)"], rows)
            + "\n\n" + table(["Train-pool", "Value"], [
                ["Label noise (all)", pct(tp["label_noise"])],
                ["Removed by agreement filter", f"{tp['removed_by_filter']:,} of {tp['n']:,} ({pct(1 - tp['kept_rate'])})"],
                ["Teacher accuracy on kept", pct(tp["teacher_accuracy_kept"])],
                ["Teacher accuracy on removed", pct(tp["teacher_accuracy_removed"])],
                ["Label noise after filter", pct(tp["label_noise_after_filter"])]]))


def student_section(R: dict, ws: Workspace) -> str:
    if "train_student" not in R:
        return ""
    s = R["train_student"]
    out = ("### Student fine-tuning\n\n"
           f"`{s['student']}` + LoRA (r = {s['lora']['r']}), {s['trainable_params'] / 1e6:.2f}M trainable of "
           f"{s['total_params'] / 1e6:.0f}M parameters, trained on {s['n_train_examples']:,} filtered teacher labels "
           f"for {s['global_steps']} steps ({num(s['train_runtime_s'], 0)} s). Best dev macro-F1 {num(s['best_dev_macro_f1'])} "
           f"at step {s['best_step']}{' (early stopped)' if s['early_stopped'] else ''}.")
    if "learning_curve" in R:
        pts = R["learning_curve"]["points"]
        out += "\n\n" + table(["Training examples", "Dev macro-F1", "Dev accuracy", "Train time (s)"],
                              [[f"{p['n']:,}{' (all)' if p.get('all') else ''}", num(p["dev_macro_f1"]),
                                pct(p["dev_accuracy"]), num(p["train_runtime_s"], 0)] for p in pts])
        out += "\n\n" + _fig(ws, "learning_curve.png", "Learning curve")
    return out


def calibration_section(R: dict, ws: Workspace) -> str:
    if "score" not in R:
        return ""
    c = R["score"]["calibration"]
    m = R["score"]["metrics"]
    return ("### Scoring and calibration\n\n"
            f"All 77 labels are scored for every input; confidence = max class probability. Temperature fitted on dev: "
            f"**T = {num(c['temperature'], 2)}**. Free greedy generation is an invalid label on "
            f"{pct(m.get('test_generation_invalid_rate'))} of test inputs and agrees with the scorer on "
            f"{pct(m.get('test_generation_agrees_with_scoring'))}.\n\n"
            + table(["ECE (15 bins)", "Before", "After"], [["dev", num(c["dev_ece_before"]), num(c["dev_ece_after"])],
                                                           ["test", num(c["test_ece_before"]), num(c["test_ece_after"])]])
            + "\n\n" + _fig(ws, "reliability.png", "Reliability diagram"))


def paired_section(R: dict, ws: Workspace) -> str:
    if "evaluate" not in R:
        return ""
    e = R["evaluate"]
    p = e["student_vs_teacher"]
    mc = p["mcnemar"]
    out = ["### Student vs teacher: paired comparison (test)",
           table(["", "Teacher right", "Teacher wrong"], [["**Student right**", f"a = {p['a']:,}", f"b = {p['b']:,}"],
                                                          ["**Student wrong**", f"c = {p['c']:,}", f"d = {p['d']:,}"]]),
           f"Δ̂ = (b − c)/n = **{signed_pp(p['delta_hat'])}**, SE = {num(p['se'], 4)}, 95% CI "
           f"[{signed_pp(p['ci_low'])}, {signed_pp(p['ci_high'])}]. McNemar χ² = {num(mc['chi2'], 2)}, "
           f"p = {num(mc['p_value'], 4)} ({mc['method']}).",
           "**Non-inferiority** (one-sided α = 0.05): switch to the student if Z = (Δ̂ + δ)/SE > z₀.₉₅.",
           table(["Margin δ", "Z", "Lower bound of Δ", "Verdict"],
                 [[f"{100 * r['margin']:g} pp", num(r["z"], 2), signed_pp(r["lower_bound"]), f"**{r['verdict']}**"]
                  for r in p["non_inferiority"]]),
           _fig(ws, "non_inferiority.png", "Non-inferiority"),
           "**Per class** (10 most frequent test classes):",
           table(["Class", "n", "b", "c", "Δ̂", "95% CI (Wald, clipped to ±100 pp)", "Decision"],
                 [[r["label"], r["n"], r["b"], r["c"], signed_pp(r["delta_hat"], 1),
                   f"[{signed_pp(max(r['ci_low'], -1.0), 1)}, {signed_pp(min(r['ci_high'], 1.0), 1)}]", r["decision"]]
                  for r in e["per_class_student_vs_teacher"]])]
    sp = e["sample_size_planner"]
    rows = [[r["assumed_delta"], f"{100 * r['margin']:g} pp", f"{r['power']:.0%}",
             f"{r['n_required']:,}" if r["n_required"] else "not reachable"] for r in sp["table"]]
    out += ["### Sample-size planner",
            f"Discordance p_d estimated from prediction disagreement on {sp['n_unlabeled']:,} **unlabeled** items: "
            f"{pct(sp['p_d_unlabeled_disagreement'])} (an upper bound). Dev estimate of Δ: {signed_pp(sp['delta_dev_estimate'])}.",
            table(["Assumed Δ", "Margin δ", "Power", "Items needed"], rows)]
    lo = e["label_only_disagreements"]
    out += ["### Label only the disagreements",
            f"Gold labels are needed only where the two models disagree: {lo['labels_needed']:,} of {lo['n']:,} test items "
            f"(**{pct(lo['labeling_saved'])} less labeling**). Δ̂ from disagreements only = {signed_pp(lo['delta_disagreements_only'])}; "
            f"Δ̂ from all labels = {signed_pp(lo['delta_full_labels'])} (identical: {lo['identical']})."]
    oc = e["other_comparisons"]
    out += ["### Is an LLM needed at all?",
            table(["Comparison", "Δ̂ (first − second)", "95% CI", "McNemar p"],
                  [[k.replace("_vs_", " vs ").replace("_", " "), signed_pp(v["delta_hat"]),
                    f"[{signed_pp(v['ci_low'])}, {signed_pp(v['ci_high'])}]", num(v["mcnemar"]["p_value"], 4)]
                   for k, v in oc.items()])]
    return "\n\n".join(x for x in out if x)


def cascade_section(R: dict, ws: Workspace) -> str:
    if "cascade" not in R:
        return ""
    c = R["cascade"]
    t, ca = c["test"], c["costs_assumed"]
    rows = [["Teacher only", pct(t["teacher_only_accuracy"]), "100.0%", num(t["teacher_only_cost"]["cost_per_1k_tickets"], 2)],
            ["Student only", pct(t["student_only_accuracy"]), "0.0%", num(t["student_only_cost"]["cost_per_1k_tickets"], 2)],
            ["Cascade", pct(t["cascade_accuracy"]), pct(t["share_sent_to_teacher"]), num(t["cascade_cost"]["cost_per_1k_tickets"], 2)]]
    thr = c["effective_threshold"]
    return ("### Cascade\n\n"
            f"Assumed costs: error E = ${ca['error_cost_E']:g}, teacher call C_big = ${ca['teacher_call_cost_C_big']:g}. "
            f"Route to the teacher when (p_big − p_small)·E > C_big (p_big per dev confidence bin), or when the student's "
            f"output is invalid ({t['routed_by_invalid_output']} items) or the input length is an outlier "
            f"({t['routed_by_length_outlier']} items). Effective confidence threshold: "
            f"{num(thr) if thr is not None else 'none (no item routed by cost)'}.\n\n"
            + table(["Policy (test)", "Accuracy", "Sent to teacher", "Cost per 1k tickets ($)"], rows)
            + f"\n\nCascade cost vs teacher-only: **{pct(t['saving_vs_teacher_only'])} saving**.\n\n"
            + _fig(ws, "coverage_accuracy.png", "Coverage vs accuracy"))


def serving_section(R: dict, ws: Workspace) -> str:
    out = []
    if "serving" in R:
        s = R["serving"]
        rows = [[r["engine"], r["concurrency"], num(r["requests_per_s"], 1), num(r["output_tokens_per_s"], 0),
                 num(1000 * r["p50_latency_s"], 0), num(1000 * r["p95_latency_s"], 0), num(1000 * r["p50_ttft_s"], 0)]
                for r in s["rows"]]
        out += [f"### Serving benchmark ({s['device']}, {s['n_requests']} requests, max {s['max_tokens']} new tokens)",
                table(["Engine", "Concurrency / batch", "Requests/s", "Tokens/s", "p50 latency (ms)", "p95 latency (ms)",
                       "p50 TTFT (ms)"], rows)]
        out += [f"_Note: {n}_" for n in s["notes"]]
        out.append(_fig(ws, "throughput_concurrency.png", "Throughput vs concurrency"))
    if "quantize" in R:
        q = R["quantize"]
        rows = []
        for mode, v in q["variants"].items():
            verdicts = ", ".join(f"δ={100 * r['margin']:g}pp: {r['verdict']}" for r in v["non_inferiority"])
            rows.append([mode, pct(v["acc_new"]), pct(v["acc_ref"]), signed_pp(v["delta_hat"]),
                         pct(v["prediction_agreement"]), num(v["footprint_mb"], 0), verdicts])
        out += ["### Is quantization safe? (paired non-inferiority vs the unquantized student, test)",
                f"Reference footprint: {num(q['reference_footprint_mb'], 0)} MB ({q['reference']}).",
                table(["Variant", "Accuracy", "Reference", "Δ̂", "Same prediction", "Size (MB)", "Verdicts"], rows)]
    return "\n\n".join(x for x in out if x)


def timings_section(ws: Workspace) -> str:
    rows = []
    for marker in sorted(ws.artifacts.glob("*/_DONE.json")):
        info = read_json(marker)
        rows.append([marker.parent.name, num(info.get("seconds"), 0)])
    return "### Stage run times (s)\n\n" + table(["Stage", "Seconds"], rows) if rows else ""


def linkedin(R: dict, cfg: Config) -> str:
    """A short post with 3-4 headline numbers, all pulled from results."""
    if not {"evaluate", "cascade"} <= R.keys():
        return ""
    e, c = R["evaluate"], R["cascade"]["test"]
    p = e["student_vs_teacher"]
    primary = next((r for r in p["non_inferiority"] if math.isclose(r["margin"], cfg.evaluation.primary_delta)), p["non_inferiority"][0])
    st, te = e["models"]["student"]["test"], e["models"]["teacher"]["test"]
    gpu = R.get("env", {}).get("environment", {}).get("gpu")
    parts = [f"I distilled `{cfg.teacher.model}` into a LoRA-tuned `{cfg.student.model}` for 77-way banking support-ticket "
             f"routing (Banking77), using only teacher labels and a single {gpu or 'CPU'}.",
             f"On {e['n_test']:,} held-out tickets the student reaches {pct(st['accuracy'])} accuracy vs the teacher's "
             f"{pct(te['accuracy'])} (Δ {signed_pp(p['delta_hat'])}, 95% CI [{signed_pp(p['ci_low'])}, {signed_pp(p['ci_high'])}]); "
             f"paired non-inferiority at a {100 * primary['margin']:g}-point margin says: {primary['verdict']}.",
             f"A calibrated confidence cascade sends {pct(c['share_sent_to_teacher'])} of tickets to the teacher at "
             f"{pct(c['cascade_accuracy'])} accuracy, {pct(c['saving_vs_teacher_only'])} cheaper than teacher-only under the stated cost assumptions."]
    if "serving" in R:
        vl = [r for r in R["serving"]["rows"] if r["engine"] == "vllm fp16"]
        if vl:
            best = max(vl, key=lambda r: r["requests_per_s"])
            parts.append(f"Served with vLLM it handles {num(best['requests_per_s'], 0)} requests/s at concurrency "
                         f"{best['concurrency']} (p95 {num(1000 * best['p95_latency_s'], 0)} ms).")
    parts.append("Every number is reproducible from the repo.")
    return "### LinkedIn summary\n\n> " + " ".join(parts)


def render(cfg: Config, ws: Workspace) -> str:
    R = load_all(ws)
    sections = [headline(R, cfg), linkedin(R, cfg), data_section(R), teacher_section(R), student_section(R, ws),
                calibration_section(R, ws), paired_section(R, ws), cascade_section(R, ws), serving_section(R, ws),
                timings_section(ws)]
    stamp = max((v.get("timestamp", "") for v in R.values()), default="")
    head = f"_Generated by `python -m llms run --stage report` from `results/{cfg.run.name}/` (latest result: {stamp})._"
    return "\n\n".join([head] + [s for s in sections if s])


def update_readme(readme: Path, body: str) -> bool:
    text = readme.read_text()
    if START not in text or END not in text:
        return False
    pre, rest = text.split(START, 1)
    post = rest.split(END, 1)[1]
    readme.write_text(f"{pre}{START}\n{body}\n{END}{post}")
    return True


def run(cfg: Config, ws: Workspace, key: str) -> dict:
    body = render(cfg, ws)
    (ws.results / "REPORT.md").write_text(f"# Results: run `{cfg.run.name}`\n\n{body}\n")
    readme = ws.base / "README.md"
    updated = cfg.run.name == "full" and readme.exists() and update_readme(readme, body)
    log.info("Wrote %s%s", ws.results / "REPORT.md", " and updated README.md" if updated else "")
    return {"readme_updated": updated}
