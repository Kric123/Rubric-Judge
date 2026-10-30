from openai import OpenAI
import json
from judges import RubricJudge,BaselineJudge
from harness import Case, run,Label
from metrics import compare, label_view
from cases import CASES
from longmem_fixture import load_cases
from longmem_public import load_public_cases

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
#base = r"C:/Users/Kyle/Desktop/Honcho/tests/unified/test_cases/"
#cases = CASES
#cases = [c for c in CASES if "idk" in c.id]
#cases = load_cases(
#    [base + "longmem_ancash.json", base + "longmem_plank.json",\
#     base + "longmem_ancash_directional.json", base + "longmem_ancash_no_session.json"],
#    client, model="qwen9b-32k",
#)
cases = load_public_cases("longmemeval_oracle.json", client, model="qwen9b-32k", n=30)


baseline = BaselineJudge(client, model="qwen9b-32k")
rubric   = RubricJudge(client, model="qwen9b-32k")

for c in cases:
    print(c.id, "->", repr(c.actual)[:200])

base = run(cases, baseline)
rub  = run(cases, rubric)
#print(json.dumps(compare(base, rub, cases), indent=2, default=str))
label_view(base, rub, cases)
for r in base + rub:
      if r.judge_error:
        print(r.case_id, "|", r.method, "|", repr(r.reasoning)[:200])