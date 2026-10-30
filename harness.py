from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Label(Enum):
    SUPPORTED = "supported"
    MISSED = "missed"
    NO_BASIS = "no_basis"
    CORRECT_IDK = "correct_idk"
    CONTRADICTED = "contradicted"


PASS_LABELS = {Label.SUPPORTED, Label.CORRECT_IDK}


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected: str
    actual: str
    context: str | None = None
    question_type: str | None = None
    human_label: Label | None = None


@dataclass(frozen=True)
class Result:
    case_id: str
    judge_error: bool
    method: str
    reasoning: str
    label: Label | None = None
    passed: bool | None = None
    confidence: float | None = None
    fallback_verdict: bool | None = None


@dataclass(frozen=True)
class Verdict:
    label: Label
    confidence: float | None = None
    reasoning: str = ""


class Judge(Protocol):
    def __call__(self, case: Case) -> Verdict: ...


def run(cases, judge: Judge) -> list[Result]:
    results = []

    for case in cases:
        try:
            verdict = judge(case)
        except Exception as e: # noqa: BLE001 - any judge failure must become a flagged result
            results.append(
                Result(
                    case_id=case.id,
                    label=None,
                    passed=None,
                    confidence=None,
                    judge_error=True,
                    method="unknown",
                    reasoning=str(e),
                    fallback_verdict=None,
                )
            )
            continue
        results.append(
            Result(
                case_id=case.id,
                label=verdict.label,
                passed=verdict.label in PASS_LABELS,
                confidence=verdict.confidence,
                judge_error=False,
                method="judge",
                reasoning=verdict.reasoning,
            )
        )

    return results
