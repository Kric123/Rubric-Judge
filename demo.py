import json  # noqa: F401 - used by the optional compare() print below
import os

from openai import OpenAI

from harness import run
from judges import BaselineJudge, RubricJudge
from metrics import compare, label_view  # noqa: F401 - compare used once cases are labeled

# --- config ---------------------------------------------------------------
MODE = "synthetic"          # "synthetic" | "fixture" | "public"
PROVIDER = "ollama"         # "ollama" | "openai" | "gemini"  (Claude: see README extension point)
HONCHO_FIXTURES = "..."  # local Honcho checkout
ORACLE_PATH = "longmemeval_oracle.json"  # download separately - see README

# Every provider below speaks the OpenAI chat-completions interface, so one client
# type covers all three. Set the matching API key env var before running a hosted one.
PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama",
               "model": "qwen3.5:9b"},  # local; use the 32k Modelfile for long contexts
    "openai": {"base_url": None, "api_key": os.environ.get("OPENAI_API_KEY"),
               "model": "gpt-4o"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "api_key": os.environ.get("GEMINI_API_KEY"), "model": "gemini-2.5-flash"},
}
# --------------------------------------------------------------------------

cfg = PROVIDERS[PROVIDER]
MODEL = cfg["model"]
client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

if MODE == "synthetic":
    from cases import CASES
    cases = CASES
elif MODE == "fixture":
    from longmem_fixture import load_cases
    cases = load_cases(
        [HONCHO_FIXTURES + "longmem_ancash.json", HONCHO_FIXTURES + "longmem_plank.json"],
        client, model=MODEL,
    )
elif MODE == "public":
    from longmem_public import load_public_cases
    cases = load_public_cases(ORACLE_PATH, client, model=MODEL, n=30, seed=0)
else:
    raise ValueError(f"unknown MODE: {MODE!r}")

baseline = run(cases, BaselineJudge(client, model=MODEL))
rubric = run(cases, RubricJudge(client, model=MODEL))

label_view(baseline, rubric, cases)

# compare() yields accuracy/kappa and the recovered-case list once cases carry
# human_label values. The synthetic cases do; the generated real-data cases don't yet.
# print(json.dumps(compare(baseline, rubric, cases), indent=2, default=str))

for r in baseline + rubric:
    if r.judge_error:
        print(r.case_id, "|", r.method, "|", repr(r.reasoning)[:200])
