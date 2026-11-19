# Six-Benchmark Evaluation & Fidelity Audit (`Section 4.1, Appendices E, G`)

Prompt materialization and gold-verified offline scorers for **GSM8K** (`n = 1,319`), **IFEval** (`n = 541`), **MATH500** (`n = 500`), **MBPP** (`n = 500`), **BBH** (`n = 6,511`), and **MMLU-Pro** (`n = 12,032`).

## Files

| Path | Description |
| :--- | :--- |
| `materialize_x.py` | Renders zero-shot chat-template prompts from Hugging Face datasets via `lm-eval` (`DumpLM`) |
| `prep_x.py` | Tokenizes chat prompts once into `{"doc_id", "ids"}` JSONL so all decoders run on identical token sequences |
| `score_tasks.py` | Scores `GSM8K` (`flexible-extract`) and `IFEval` (`prompt_level_strict_acc` with seeded `langdetect`) |
| `score_x.py` | Scores `MATH500` (`math_verify` with Answer-line extraction), `MBPP` (`pass@1` with restored fenced prefix), `BBH`, and `MMLU-Pro`, with `--selftest` validation on gold answers |
| `lmo_common.py` | Lightweight `DumpLM` / `ReplayLM` `lm-eval` wrappers |
| `data_meta/` | Prompt provenance & SHA256 metadata across all six benchmarks |
| `results/` | Full-test-set summaries (`RESULTS_FULL.json`, `RESULTS_FULL.md`) and teacher-forced fidelity audit (`audit_aggregate.json`) |

## Quickstart

```bash
# 1. Materialize and tokenize benchmark prompts
python eval/prep_x.py --model Xalphinions/ouro-2.6b-late-entry --task gsm8k
python eval/materialize_x.py \
  --model Xalphinions/ouro-2.6b-late-entry \
  --tasks minerva_math500,mbpp_instruct,bbh_cot_zeroshot,mmlu_pro

# 2. Self-test gold answer extractors before scoring model outputs
python eval/score_x.py --task math500 --model Xalphinions/ouro-2.6b-late-entry --selftest
python eval/score_x.py --task mbpp --model Xalphinions/ouro-2.6b-late-entry --selftest

# 3. Score decoded JSON outputs
python eval/score_tasks.py --task gsm8k --in results/ouro_latek_gsm8k.json --out results/score_gsm8k.json
python eval/score_x.py --task math500 --model Xalphinions/ouro-2.6b-late-entry --glob "results/math500_*.json" --out results/score_math500.json
```
