# rubric-judge

A diagnostic LLM judge for conversational memory systems. A normal judge returns a
pass/fail. This one labels why an answer failed, and it never lets a judge error turn into
a silent pass or fail.

A binary judge tells you whether an answer was right. It doesn't tell you why it was wrong,
and some answers that are wrong for opposite reasons look identical to it. This harness
runs a binary judge (the current standard) next to a five-label rubric judge on the same
cases, so you can see where they diverge.

> Scope: this tests the judge, not a production system. Every answer under test is
> generated locally with a small open model (Ollama, `qwen3.5:9b`). The runs below are
> illustrative, on synthetic and public-benchmark data. They are not a measurement of any
> memory product.

## Two problems it targets

**Pass/fail throws away the reason.** For a memory system, the reason an answer failed is
the useful part. Retrieval might have failed to surface a fact (a recall bug), or
generation might have made something up (a hallucination). Those are different problems
with different fixes. A yes/no judge turns both into one "FAIL".

**A judge error can turn into a score.** In Honcho's benchmark judge, when the grading call
throws, the code logs the error, falls back to substring matching, and returns that as the
pass/fail. It still counts in the aggregate success rate as a normal result:

- [`tests/bench/longmem_common.py#L288-L295`](https://github.com/plastic-labs/honcho/blob/64a78ae812448cda60efbfaf0846bbf154468d31/tests/bench/longmem_common.py#L288-L295)
- [`tests/bench/locomo_common.py#L448-L457`](https://github.com/plastic-labs/honcho/blob/64a78ae812448cda60efbfaf0846bbf154468d31/tests/bench/locomo_common.py#L448-L457)

The error shows up in the logs, but the headline number treats a degraded verdict the same
as a real one. This harness treats a judge error as its own outcome. It gets flagged, left
out of the accuracy denominator, and counted on its own, so it can't quietly move the
score. Permalinks are pinned at Honcho `64a78ae` and used for reference. This is a
standalone, clean-room project.

## The labels

The rubric judge returns one of five labels. Each is a rule about the answer relative to
what the conversation actually supports, not world knowledge.

| Label | The answer | Maps to |
|---|---|---|
| `supported` | states the answer, and the conversation backs it up | correct |
| `correct_idk` | correctly declines, because the info isn't in the conversation | correct |
| `missed` | the fact is in the conversation, but the model didn't give it (wrong, vague, or wrongly declined) | retrieval / recall failure |
| `no_basis` | asserts something the conversation never established | unfounded generation |
| `contradicted` | states something the conversation directly contradicts | hallucinated fact |

`PASS_LABELS = {supported, correct_idk}`. An answer passes only when it's correct, and
correctly saying "I don't know" counts as correct.

Two cases a binary judge can't separate:

- `missed` vs `correct_idk`. The same answer, "I don't have that information," is right when
  the fact isn't in the conversation (`correct_idk`) and wrong when the fact is there and
  the model missed it (`missed`). You need the conversation to tell them apart.
- `contradicted` vs `no_basis`. Both are wrong claims. The split is whether the conversation
  says nothing about the claim (`no_basis`) or says the opposite (`contradicted`). That
  lines up with a retrieval gap versus a generation error.

## Results

Two runs. The synthetic set is built to cover all five labels. The oracle set shows the
same labels showing up on a benchmark people actually use. In both, the judge is a local
`qwen3.5:9b` and the answers under test are generated locally.

### Synthetic cases (`demo.py`, `MODE="synthetic"`)

Eight hand-written cases over two short conversations, each with a hand-assigned label
(`cases.py`).

| case | ground truth | binary | rubric |
|---|---|---|---|
| `move-supported` | supported | PASS | supported |
| `move-contradicted` | contradicted | FAIL | contradicted |
| `move-missed` | missed | FAIL | missed |
| `move-nobasis` | no_basis | FAIL | contradicted |
| `move-idk` | correct_idk | PASS | correct_idk |
| `job-supported` | supported | PASS | supported |
| `job-contradicted` | contradicted | FAIL | contradicted |
| `job-idk` | correct_idk | PASS | correct_idk |

The two judges agree on the four correct answers and on both abstentions. The difference is
on the four FAILs: the binary judge only says FAIL, while the rubric names the reason, here
three `contradicted` and one `missed`. `move-nobasis` is the one row where the rubric and my
hand label disagree. I called it `no_basis`; the rubric called it `contradicted`. The
conversation says the pet is a cat, so answering "a dog" contradicts it rather than being
unfounded, which is a reasonable read and the kind of boundary a human-labeled set would
settle (see roadmap). Both abstention cases also pass under the binary judge, because here
it reads the conversation too. The case where the binary judge fails a correct decline shows
up in the oracle run below (`35a27287`).

### Public LongMemEval oracle sample (`demo.py`, `MODE="public"`, n=30, seed=0)

A 30-question sample of the public [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
oracle set (evidence sessions only), about 30% abstention questions. The binary judge
scored 24 pass and 6 fail. The rubric breaks those down:

| rubric label | n | what it tells you |
|---|---|---|
| `supported` | 20 | correct, grounded answers |
| `correct_idk` | 5 | correct abstentions |
| `missed` | 4 | answer was present; model gave empty, wrong, or a wrong decline |
| `contradicted` | 1 | model stated facts not in the conversation |

Selected rows:

| case | binary | rubric | what the rubric adds |
|---|---|---|---|
| `09ba9854_abs` | FAIL | `contradicted` | abstention question; model invented airport transit costs, a generation bug |
| `gpt4_4fc4f797` | FAIL | `missed` | answer was in the conversation; model wrongly said "I don't have that", a recall bug |
| `36b9f61e` | FAIL | `missed` | same failure mode |
| `gpt4_78cf46a3` | FAIL | `missed` | empty answer |
| `35a27287` | FAIL | `correct_idk` | model correctly declined (no location given); binary fails it anyway |
| `f685340e_abs` | PASS | `correct_idk` | correct abstention, now labeled as one |
| `22d2cb42` | PASS | `supported` | correct, grounded answer |

The first four rows are the point. The binary judge reports four identical FAILs. One is a
hallucination (`contradicted`) and three are recall failures (`missed`), which point at
different parts of the system. `35a27287` is a recovery: a correct decline that the binary
judge still fails. The two-sided abstention case is here in real data, same surface string
with opposite verdicts. `gpt4_4fc4f797` says "I don't have that information" when the answer
was in the conversation (`missed`), while `f685340e_abs` says the same thing when it
genuinely wasn't (`correct_idk`).

### A limitation this run exposed

The 9B judge labeled 4 correct abstentions as `supported` when they should have been
`correct_idk` (`gpt4_70e84552_abs`, `80ec1f4f_abs`, `e5ba910e_abs`, `edced276_abs`). Each
was a hedged abstention, like "I don't have that information; the conversation mentions X
but not Y," and the judge keyed on the "mentions X" part. Abstention detection at this model
size isn't reliable. A larger judge (see Providers) or a separate abstention check would
help, and it's on the roadmap. A binary judge can't make this distinction at all, so this
is a limit on how precise the rubric gets.

## Design

- Thin judges, one place for errors. A judge returns a `Verdict` or raises. The harness
  wraps each case in a single try/except (`harness.py`, `run`) and turns a failure into a
  flagged `Result` with `judge_error=True`, so a failure is recorded rather than swallowed.
- `Judge` is a `Protocol`. `BaselineJudge` and `RubricJudge` are interchangeable. A new
  judge is another callable `(Case) -> Verdict`.
- Both judges read the conversation. The rubric needs it for its labels, and the baseline
  gets it too so the comparison is fair.
- Same-denominator scoring. `metrics.compare` scores both judges only on the cases both
  judged without error (`common_ids`), which avoids survivorship bias, and reports the
  error counts separately so the cost of a fragile judge stays visible.
- Kappa is `None` when it's undefined. On tiny or single-class label sets Cohen's kappa has
  no meaningful value, so the code returns `None` instead of a number. Accuracy and kappa
  come into play once cases have hand-applied `human_label`s.

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

- `synthetic`: self-contained, no download.
- `fixture`: runs on Honcho's own LongMemEval unit-test fixtures from a local Honcho
  checkout. Point `HONCHO_FIXTURES` at that checkout's `tests/unified/test_cases/` directory,
  either as an environment variable or by editing the constant in `demo.py`:

  ```bash
  export HONCHO_FIXTURES=/path/to/honcho/tests/unified/test_cases/
  ```
- `public`: download the LongMemEval oracle file to `longmemeval_oracle.json` (it's
  gitignored, not vendored here). Source:
  [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/main/longmemeval_oracle.json)
  on HuggingFace. Direct download:

  ```bash
  curl -L -o longmemeval_oracle.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
  ```

## Providers

The judges call the model through the OpenAI chat-completions interface, so the same code
runs against anything that speaks it. Set `PROVIDER` in `demo.py`:

| `PROVIDER` | Needs | Notes |
|---|---|---|
| `ollama` | local Ollama | default; no API key, no GPU needed for the small sets |
| `openai` | `OPENAI_API_KEY` | native OpenAI endpoint |
| `gemini` | `GEMINI_API_KEY` | Google's OpenAI-compatible endpoint |

Claude isn't a drop-in. The Anthropic SDK (`client.messages.create`, content blocks) has a
different shape than chat-completions. Adding it means a small `Completer` adapter behind
the judges, one method `complete(prompt) -> str`, with an Anthropic-backed version. The
`Judge` protocol is already the right place for that. It's on the roadmap.

## Limitations and roadmap

- Judge size. `qwen3.5:9b` slips on hedged abstentions (above). A larger hosted judge
  (OpenAI or Gemini today, Claude via the adapter above), or a dedicated abstention
  classifier, should lift `correct_idk` precision.
- Native Claude provider. Add the `Completer` seam so the Anthropic SDK is a first-class
  judge backend alongside the OpenAI-compatible providers.
- No human-labeled set yet. Accuracy and kappa are wired up but need hand-applied
  `human_label`s on a real subset to mean anything. That's the next step.
- Oracle-scale only. Runs use the oracle (evidence-only) split. Full-haystack LongMemEval
  (~115K tokens) is future work.
- No confidence calibration or cost and latency accounting yet.

## What this is and isn't

- Standalone and clean-room (MIT). It references Honcho by permalink and copies none of its
  code.
- The `actual` answers are generated locally by a small open model to give the judge
  realistic text to grade. They are not output from Honcho or any other memory system, so
  this isn't an audit of a product.
- Synthetic data is labeled as synthetic. Benchmark data is downloaded, not redistributed.
