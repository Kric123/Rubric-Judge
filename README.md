# rubric-judge

A diagnostic LLM-judge harness for conversational memory systems. Instead of a binary
pass/fail, it labels **why** an answer failed — and never lets a judge error quietly
count as a pass or fail.

A binary judge tells you *whether* an answer was right. It can't tell you *why* it was
wrong, and the two wrong-for-opposite-reasons cases below look identical to it. This
harness runs a binary judge (the status quo) and a five-label rubric judge side by side
so the difference is visible on the same cases.

> **Scope.** This tests the *judge*, not any production system. Every answer-under-test
> is generated locally with a small open model (Ollama / `qwen3.5:9b`). The numbers here
> are illustrative runs on synthetic and public-benchmark data, not a measurement or audit
> of any memory product.

## The two problems it targets

**1. "Right or wrong" throws away the reason.** For a memory system, *why* an answer
failed is the whole signal: did retrieval fail to surface the fact (a recall bug), or did
generation make something up (a hallucination)? Those need different fixes. A yes/no judge
collapses them into one "FAIL."

**2. A judge error can silently become a score.** In Honcho's benchmark judge, if the
grading model call throws, the code logs the error and falls back to substring matching,
then returns that as the pass/fail — which still counts in the aggregate success rate as
an ordinary result:

- [`tests/bench/longmem_common.py#L288-L295`](https://github.com/plastic-labs/honcho/blob/64a78ae812448cda60efbfaf0846bbf154468d31/tests/bench/longmem_common.py#L288-L295)
- [`tests/bench/locomo_common.py#L448-L457`](https://github.com/plastic-labs/honcho/blob/64a78ae812448cda60efbfaf0846bbf154468d31/tests/bench/locomo_common.py#L448-L457)

The error is logged, but the top-line number can't tell a real verdict from a degraded
one. This harness makes a judge error a **first-class, flagged outcome**: it's excluded
from the accuracy denominator and reported as its own count, so it can never inflate or
deflate the score unnoticed. (Permalinks pinned at Honcho `64a78ae`; referenced for
context only — this is a clean-room, standalone project.)

## The labels

The rubric judge returns one of five labels, each a decision rule relative to **what the
conversation actually supports** (not world knowledge):

| Label | The answer… | Maps to |
|---|---|---|
| `supported` | states the answer, and the conversation backs it up | correct |
| `correct_idk` | correctly declines — the info genuinely isn't in the conversation | correct |
| `missed` | the fact **is** in the conversation, but the model didn't give it (wrong, vague, or wrongly abstained) | retrieval / recall failure |
| `no_basis` | asserts something the conversation never established | unfounded generation |
| `contradicted` | states something the conversation **directly contradicts** | hallucinated fact |

`PASS_LABELS = {supported, correct_idk}` — an answer passes only when it's *correct*,
which includes correctly saying "I don't know."

Two distinctions a binary judge structurally cannot make:

- **`missed` vs `correct_idk`** — the *same* surface answer, "I don't have that
  information," is **right** when the fact is absent (`correct_idk`) and **wrong** when the
  fact is present but the model failed to recall it (`missed`). Only grounding in the
  conversation separates them.
- **`contradicted` vs `no_basis`** — both are wrong assertions, split by whether the
  conversation is *silent* on the claim (`no_basis`) or *says the opposite*
  (`contradicted`) — i.e. a retrieval gap vs a generation error.

## Results

Two illustrative runs. The synthetic set exercises the full label range by construction;
the oracle set shows the same labels emerging on data from a benchmark the field actually
uses. Judge = local `qwen3.5:9b`; answers-under-test also generated locally.

### Synthetic, conversation-grounded cases (`demo.py`, `MODE="synthetic"`)

Eight hand-written cases over two short conversations, each with a known ground-truth label
(`cases.py`). They span all five labels and include the two-sided abstention pair:

| case | ground truth | binary | rubric |
|---|---|---|---|
| `move-supported` | supported | PASS | supported |
| `move-contradicted` | contradicted | FAIL | contradicted |
| `move-missed` | missed | FAIL | missed |
| `move-nobasis` | no_basis | FAIL | no_basis |
| `move-idk` | correct_idk | PASS | correct_idk |
| `job-supported` | supported | PASS | supported |
| `job-contradicted` | contradicted | FAIL | contradicted |
| `job-idk` | correct_idk | **FAIL** | **correct_idk** |

`job-idk` is the recovery: the model correctly declines a question the conversation can't
answer, the binary judge marks it **FAIL**, and the rubric correctly labels it
`correct_idk`. Same surface string as `move-idk`, opposite correct verdict.

### Public LongMemEval oracle sample (`demo.py`, `MODE="public"`, n=30, seed=0)

A 30-question sample of the public [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
oracle set (evidence sessions only), ~30% abstention questions. The binary judge scored
**24 pass / 6 fail**; the rubric split those failures — and relabeled the passes — into:

| rubric label | n | what it tells you |
|---|---|---|
| `supported` | 20 | correct, grounded answers |
| `correct_idk` | 5 | correct abstentions |
| `missed` | 4 | answer was present; model gave empty/wrong/wrongly-abstained |
| `contradicted` | 1 | model fabricated facts not in the conversation |

Selected rows (binary verdict vs rubric diagnosis):

| case | binary | rubric | what the rubric adds |
|---|---|---|---|
| `09ba9854_abs` | FAIL | `contradicted` | abstention question; model **invented** airport transit costs → a generation bug |
| `gpt4_4fc4f797` | FAIL | `missed` | answer was in the conversation; model wrongly said "I don't have that" → a recall bug |
| `36b9f61e` | FAIL | `missed` | same failure mode as above |
| `gpt4_78cf46a3` | FAIL | `missed` | empty answer |
| `35a27287` | **FAIL** | `correct_idk` | model correctly declined (no location given); binary wrongly fails it |
| `f685340e_abs` | PASS | `correct_idk` | correct abstention, now labeled as one |
| `22d2cb42` | PASS | `supported` | correct, grounded answer |

The payoff is in the first four rows: the binary judge reports four identical "FAIL"s,
but one is a **hallucination** (`contradicted`) and three are **recall failures**
(`missed`) — different bugs, different owners. And `35a27287` is the same abstention
recovery the synthetic set shows, on real data.

### Honest limitation from this run

The 9B judge mislabeled **4 correct abstentions as `supported`** instead of `correct_idk`
(`gpt4_70e84552_abs`, `80ec1f4f_abs`, `e5ba910e_abs`, `edced276_abs`). Each was a
*hedged* abstention — "I don't have that information; the conversation mentions X but not
Y" — and the judge latched onto the "mentions X" clause. The rubric's abstention detection
is imperfect at this model size; a stronger judge (see Providers) or an explicit two-stage
abstention check is on the roadmap. A binary judge can't see this distinction at all, so
it's a ceiling on the rubric's precision, not a regression against the baseline.

## Design

- **Thin judges, central error handling.** A judge either returns a `Verdict` or raises.
  The harness owns one `try/except` per case (`harness.py:run`) and turns any failure into
  a flagged `Result(judge_error=True)` — errors are data, not crashes and not silent passes.
- **`Judge` is a `Protocol`.** `BaselineJudge` and `RubricJudge` are interchangeable; a new
  judge is just another callable `(Case) -> Verdict`.
- **Both judges read the conversation.** The rubric's labels require it; the baseline gets
  it too, so the comparison is fair — neither judge is handicapped on inputs.
- **Fair, same-denominator scoring.** `metrics.compare` scores both judges on only the
  cases *both* judged without error (`common_ids`), so no survivorship bias, and reports
  the error counts separately so JSON-fragility cost stays visible.
- **Kappa is `None`, not faked, when undefined.** On tiny or single-class label sets
  Cohen's kappa is undefined; the code returns `None` rather than a misleading number.
  Accuracy/kappa activate once cases carry hand-applied `human_label`s.

## Running it

Requires [Ollama](https://ollama.com) (for the local default) and the `openai` Python
client.

```bash
pip install openai scikit-learn
ollama pull qwen3.5:9b
# long conversations need a larger context window than Ollama's 4K default:
ollama create qwen9b-32k -f Modelfile        # FROM qwen3.5:9b, num_ctx 32768
```

Pick a data source in `demo.py` (`MODE = "synthetic" | "fixture" | "public"`) and run:

```bash
python demo.py
```

- **synthetic** — self-contained; no download.
- **fixture** — reads Honcho's own LongMemEval unit-test fixtures from a local Honcho
  checkout (set `HONCHO_FIXTURES`).
- **public** — download the LongMemEval oracle file to `longmemeval_oracle.json`
  (from [the LongMemEval repo](https://github.com/xiaowu0162/LongMemEval); it's gitignored,
  not vendored here).

### Providers

The judges talk to the model through the OpenAI chat-completions interface, so the same
code runs against any provider that speaks it. Set `PROVIDER` in `demo.py`:

| `PROVIDER` | Needs | Notes |
|---|---|---|
| `ollama` | local Ollama | default; no API key, no GPU required for the small sets |
| `openai` | `OPENAI_API_KEY` | uses the native OpenAI endpoint |
| `gemini` | `GEMINI_API_KEY` | via Google's OpenAI-compatible endpoint |

**Claude** is a clean extension point rather than a drop-in: the Anthropic SDK
(`client.messages.create`, content blocks) has a different shape than chat-completions, so
adding it means a small `Completer` adapter behind the judges (one method,
`complete(prompt) -> str`) with an Anthropic-backed implementation — the `Judge` protocol
is already the right seam for it. Noted on the roadmap below.

## Limitations & roadmap

- **Judge size.** `qwen3.5:9b` slips on hedged abstentions (above). A larger hosted judge
  (OpenAI/Gemini today, Claude via the adapter above), or a dedicated abstention
  classifier, should lift `correct_idk` precision.
- **Native Claude provider.** Add the `Completer` seam so the Anthropic SDK is a
  first-class judge backend alongside the OpenAI-compatible providers.
- **No human-labeled set yet.** Accuracy and kappa are wired up but need hand-applied
  `human_label`s on a real subset to be meaningful; that's the next step.
- **Oracle-scale only.** Runs use the oracle (evidence-only) split; full-haystack
  LongMemEval (~115K tokens) is future work.
- **No confidence calibration or cost/latency accounting** yet.

## What this is / isn't

- It is a **standalone, clean-room** project (MIT). It references Honcho only by permalink
  and copies none of its code.
- The `actual` answers are **locally generated** by a small open model to stress the judge
  on realistic text — they are **not** the output of Honcho or any other memory system, so
  nothing here is an audit of a product.
- Synthetic data is labeled as synthetic; benchmark data is downloaded, not redistributed.
