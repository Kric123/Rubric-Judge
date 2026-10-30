from harness import Case, Result, Label, PASS_LABELS
from sklearn.metrics import cohen_kappa_score as cohen_kappa

def is_passed(label: Label |None) -> bool:

    return label in PASS_LABELS

def score_judge(results: list[Result],cases: list[Case], only_ids= None) -> dict:
    by_id = {case.id: case for case in cases}
    human, judges = [],[]
    judge_errors = 0
    for result in results:
        if only_ids is not None and result.case_id not in only_ids:
            continue
        if result.judge_error:
            judge_errors += 1
            continue
        case= by_id[result.case_id]
        if case.human_label is not None:
            human.append(is_passed(case.human_label))
            judges.append(result.passed)
    accuracy = sum(1 for h,j in zip(human,judges) if h==j)/len(judges) if judges else 0.0
    if len(set(human)) > 1 and len(set(judges)) > 1:
        kappa = cohen_kappa(human, judges)
    else:
        kappa = None   # undefined: not enough label variance to compute
    return {"accuracy": accuracy, "judge_errors": judge_errors,\
             "n_scored": len(human), "kappa": kappa}

def common_ids(baseline_results, rubric_results):
    base_ok = {r.case_id for r in baseline_results if not r.judge_error}
    rub_ok  = {r.case_id for r in rubric_results  if not r.judge_error}
    return base_ok & rub_ok

def compare(baseline_results,rubric_results, cases) -> dict:
    ids = common_ids(baseline_results, rubric_results)
    base = score_judge(baseline_results, cases, only_ids=ids)
    rub = score_judge(rubric_results, cases, only_ids=ids)
    base_by_id = {r.case_id: r for r in baseline_results}
    rub_by_id  = {r.case_id: r for r in rubric_results}
    recovered = []
    for c in cases:
        if c.human_label is None:
            continue
        human_pass = is_passed(c.human_label)
        b, r = base_by_id[c.id], rub_by_id[c.id]
        if b.judge_error or r.judge_error:
            continue
        if b.passed != human_pass and r.passed == human_pass:   # baseline wrong, rubric right
            recovered.append(c.id)
    return {
        "baseline": base,
        "rubric": rub,
        "errors": {
            "baseline": sum(r.judge_error for r in baseline_results),
            "rubric":   sum(r.judge_error for r in rubric_results),
        },
        "accuracy_gain": rub["accuracy"] - base["accuracy"],
        "kappa_gain":  (rub["kappa"] - base["kappa"]) if rub["kappa"] is not \
            None and base["kappa"] is not None else None,
        "recovered": recovered,
    }


def label_view(baseline_results, rubric_results, cases) -> None:
    """Per case: binary's flat pass/fail vs the rubric's specific diagnosis."""
    base = {r.case_id: r for r in baseline_results}
    rub  = {r.case_id: r for r in rubric_results}
    print(f"{'case':6} {'human':14} {'binary':8} {'rubric (diagnosis)':20}")
    for c in cases:
        b, r = base[c.id], rub[c.id]
        human  = c.human_label.value if c.human_label else "-"
        binary = "ERROR" if b.judge_error else ("PASS" if b.passed else "FAIL")
        rubric = "JUDGE_ERROR" if r.judge_error else r.label.value
        print(f"{c.id:6} {human:14} {binary:8} {rubric:20}")


  