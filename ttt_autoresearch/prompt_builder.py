from __future__ import annotations

def build_rollout_prompt(
    *,
    state_ctx: str,
    construction_section: str,
    code_section: str,
) -> str:
    return f"""You are an expert machine learning researcher and systems engineer optimizing a language-model training script.

Your task is to improve `train.py` so that it achieves a lower `val_bpb` under the fixed AutoResearch evaluation budget.

## Problem

Improve the `train.py` program so that the resulting training run achieves a lower validation bits-per-byte (`val_bpb`).

Everything in `train.py` is fair game:
- architecture
- optimizer
- hyperparameters
- training loop
- batch size
- model size

**Lower `val_bpb` values are better** - they indicate a stronger model under the fixed evaluation budget.

## Budget & Resources
- **Time budget**: 5 minutes of wall-clock training time
- **Evaluation harness**: fixed AutoResearch runner
- **VRAM**: moderate increases are acceptable for meaningful gains, but avoid wasteful blowups

## AutoResearch Invariants
- `prepare.py` and the evaluation protocol are fixed and cannot be changed
- Maximum sequence length is `2048`
- Validation uses the pinned shard `06542`
- The tokenizer / vocabulary setup is fixed at vocab size `8192`
- The training script must remain compatible with the existing BOS-aligned bin-packing data pipeline
- The model implementation must continue to support `forward(x, y, reduction='none')`

## Rules
- You may only edit `train.py`
- Do not modify `prepare.py`, dependencies, or the evaluation harness
- Return exactly one ```json``` block with this schema:
{{
  "summary": "short description of the change",
  "rationale": "why this should improve val_bpb",
  "train_py": "the full replacement contents of train.py"
}}
- `train_py` must be the full file, not a diff
- Propose exactly one candidate for this rollout
- Optimize for the lowest `val_bpb` under the fixed time budget
- Prefer simpler changes when improvement is similar

{state_ctx}
{construction_section}
{code_section}
"""
