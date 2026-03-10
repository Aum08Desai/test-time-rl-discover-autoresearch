# Test Time RL Discover + Auto Research

![teaser](progress.png)

This repo is a focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that replaces the ad hoc outer experimentation loop with [TTT-Discover](https://github.com/test-time-training/discover).

The core idea is:

- The **inner loop** is still `autoresearch`: edit `train.py`, run a fixed-budget training job, measure `val_bpb`.
- The **outer loop** is now **test-time RL** from TTT-Discover.
- The outer model proposes full replacements for `train.py`.
- The resulting inner-loop metric improvement becomes the reward used to update the outer model online.

This keeps the original spirit of autoresearch, but makes the search policy itself train during the run.

## Credits

This project is derived from:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- [test-time-training/discover](https://github.com/test-time-training/discover)

The RL optimization recipe is intended to stay with upstream `discover`; this repo mainly provides the autoresearch-specific environment, reward, runner, and usage wrapper.

## What This Repo Does

The repo has two layers:

1. **Inner optimization target**
   - `prepare.py` downloads data and trains the tokenizer.
   - `train.py` is the only file the outer model edits.
   - `val_bpb` is the optimization metric. Lower is better.

2. **Outer TTT-Discover loop**
   - `run_ttt_discover.py` launches the test-time RL run.
   - `ttt_autoresearch/` adapts autoresearch to the `discover` environment interface.
   - Each candidate `train.py` is executed in an isolated workspace.
   - Reward is computed from `current_best_val_bpb - candidate_val_bpb`.

## Repository Layout

```text
prepare.py                  Fixed data prep and runtime utilities
train.py                    Inner training program edited by the outer model
program.md                  Human-authored research instructions/context
run_ttt_discover.py         Main TTT-Discover entrypoint
ttt_autoresearch/           Adapter layer for environment, reward, runner, config
configs/                    Ready-to-run YAML config
tests/                      Smoke and unit coverage for the adapter
```

## How The RL Loop Works

At each outer-loop step:

1. TTT-Discover samples a group of candidate `train.py` replacements.
2. Each candidate is evaluated by running a real autoresearch training job.
3. The resulting `val_bpb` is parsed from the run logs.
4. Reward is computed from improvement over the current best state.
5. Upstream `discover` performs the online RL update.
6. If a candidate improves `val_bpb`, it becomes the new best `train.py`.

Important details:

- The **action** is the full replacement contents of `train.py`.
- The **reward** is the inner-loop metric outcome, not the patch text.
- The implementation keeps grouped rollouts for the upstream entropic advantage recipe.
- The default config is now paper-shaped: `8 groups x 8 rollouts x 50 steps`.
- In this repo, groups are controlled by `groups_per_step` and rollouts within each group are controlled by `samples_per_step`.
- The checked-in default keeps `max_concurrent_evaluations: 1` for safety; to scale on rented hardware, you raise concurrency and declare explicit `gpu_devices`.

## Quick Start

**Requirements**

- Linux
- NVIDIA GPUs
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

Install and prepare the base autoresearch environment:

```bash
# 1. Install dependencies
uv sync

# 2. Download data and train the tokenizer
uv run prepare.py

# 3. Sanity check the original inner loop
uv run train.py
```

Then launch the outer TTT-Discover loop:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

## Default Configuration

The default config lives at [configs/ttt_discover_autoresearch.yaml](configs/ttt_discover_autoresearch.yaml).

Current defaults:

- `model_name: Qwen/Qwen3.5-35B-A3B`
- `groups_per_step: 8`
- `samples_per_step: 8`
- `max_steps: 50`
- `temperature: 1.0`
- `max_concurrent_evaluations: 1`

That means the default run is:

- `8 groups`
- `8 rollouts per group`
- `64 total inner evaluations per step`
- `50 outer RL steps`
- but only `1` inner evaluation runs at a time unless you explicitly provision more GPUs

This keeps the paper-shaped RL structure while remaining safe to launch on limited hardware.

## Recommended Hardware

If your goal is to match the spirit of the original autoresearch setup and push toward the best `val_bpb`, the inner loop should run on **H100 80GB** class GPUs.

Why:

- [train.py](/Users/aumdesai/AutoResearch-Discover/train.py) uses Hopper-specific FA3 kernels when available.
- [program.md](/Users/aumdesai/AutoResearch-Discover/program.md) shows representative peak VRAM around `45 GB`.
- `A100 40GB` is therefore not sufficient.

Recommended inner-loop rental target:

- **Best cost/performance:** H100 PCIe 80GB
- **Best absolute performance:** H100 SXM 80GB

For the paper-shaped default config, the natural operational target is:

- **64 H100 80GB GPUs** for inner evaluations
- one rollout per GPU
- one full outer step in roughly one inner-training wave

If you have fewer GPUs, the run still works, but each outer step takes multiple waves.
To use more than one GPU safely, you should set:

```yaml
max_concurrent_evaluations: 64
gpu_devices: ["0", "1", "2", "3", "..."]
```

The runner now pins each candidate subprocess to one configured `CUDA_VISIBLE_DEVICES` slot.

## Cost Model

There are two separate cost buckets:

1. **Inner-loop GPU rental**
   - pays for the actual `train.py` runs
   - this dominates total cost in this repo

2. **Outer-loop Tinker cost**
   - pays for model prefill, sampling, and RL training tokens
   - this is comparatively small here because the inner rollouts are expensive

### Tinker Cost

Using the official Tinker pricing for `Qwen/Qwen3.5-35B-A3B`:

- prefill: `$0.36 / 1M tokens`
- sample: `$0.89 / 1M tokens`
- train: `$1.07 / 1M tokens`

And using this repo's actual prompt/output sizes, a practical estimate is:

- about **`$0.017-$0.024` per rollout**
- about **`$0.020` per rollout** as a reasonable midpoint

So for the default paper-shaped config:

- `8 x 8 x 50 = 3200 total rollouts`
- estimated Tinker cost: about **`$54-$77`**
- midpoint estimate: about **`$65`**

### GPU Rental Cost

Using H100 PCIe 80GB pricing of roughly **`$2.86 / GPU / hour`**, and assuming one inner rollout takes roughly `325.9s` end to end:

- each rollout costs about **`$0.259`** in GPU rental
- `3200` rollouts costs about **`$829`** in GPU rental

That means a fully provisioned `8 x 8 x 50` run is roughly:

- **GPU rental:** about `$829`
- **Tinker:** about `$65`
- **Total:** about **`$894`**

This is directionally consistent with the TTT-Discover paper's statement that runs cost a few hundred dollars to several hundred dollars per problem, with this repo skewing more expensive on the inner loop because each rollout is a real GPU training job.

### Cost Distribution

For this repo, the cost split is roughly:

- **~90% GPU rental**
- **~10% Tinker**

That is the opposite of many lightweight code-generation settings. Here, the expensive part is the real autoresearch evaluation.

## How I Recommend Running It

### If you want the paper-shaped run

Use the paper-shaped structure and rent:

- **64x H100 PCIe 80GB**

Set:

```yaml
groups_per_step: 8
samples_per_step: 8
max_steps: 50
max_concurrent_evaluations: 64
gpu_devices: ["0", "1", "2", ..., "63"]
```

This gives:

- `8 groups x 8 rollouts`
- one GPU per rollout
- about one rollout wave per step
- wall-clock of roughly `50 x 5.4 minutes`, plus overhead

This is the closest clean operational match to the repo default.

### If you want a cheaper but still strong run

Use:

- `groups_per_step: 8`
- `samples_per_step: 8`
- `max_steps: 8` to `16`
- `max_concurrent_evaluations` equal to however many GPUs you actually rented

This preserves the paper-like group structure while cutting total spend materially.

### If you only have one GPU

The checked-in config is already safe in the sense that it runs with one evaluation slot, but it will be extremely slow at full `8 x 8 x 50`.

Instead reduce to something like:

```yaml
groups_per_step: 1
samples_per_step: 8
max_steps: 8
max_concurrent_evaluations: 1
gpu_devices: null
```

That is much slower and less faithful to the paper, but operationally sane on one machine.

## Model and Renderer Configuration

The model is configurable, but the prompt/response format must match a supported renderer.

Known-good renderer values:

- `qwen3`
- `qwen3_instruct`
- `gpt_oss_no_sysprompt`
- `gpt_oss_low_reasoning`
- `gpt_oss_medium_reasoning`
- `gpt_oss_high_reasoning`

Examples:

```yaml
model_name: Qwen/Qwen3.5-35B-A3B
renderer_name: qwen3
```

```yaml
model_name: openai/gpt-oss-120b
renderer_name: gpt_oss_high_reasoning
```

If you use an unknown model family, you should set `renderer_name` explicitly. The config now fails fast if it cannot infer a compatible renderer.

## Output Artifacts

Each run writes artifacts under `runs/<timestamp>/`:

- `baseline.json`
  - baseline execution metadata for the original `train.py`
- `resolved_config.json`
  - the fully resolved runtime config
- `history.jsonl`
  - one line per evaluated candidate
- `best/train.py`
  - the current best discovered inner-loop program
- `best/metrics.json`
  - the best run metadata and metric
- `candidates/`
  - isolated workspaces with stdout/stderr and per-candidate files
- `discover_log/`
  - upstream sampler/checkpoint/log state from `ttt-discover`

## Inner Loop Assumptions

This repo intentionally keeps the inner autoresearch target small even though the outer RL setup can be large:

- `prepare.py` remains fixed.
- `train.py` is the only file the outer model edits.
- Training still uses the original fixed wall-clock budget from autoresearch.
- `val_bpb` remains the optimization target because it is stable across vocabulary and architecture changes.

## Design Choices

### Why only `train.py`?

Because that matches the original autoresearch framing and keeps the action space bounded. It also makes it easier to attribute reward to specific inner-loop changes.

### Why grouped rollouts?

Because upstream `discover` uses grouped rollouts for its entropic advantage estimation and reuse behavior. This repo keeps that outer-loop recipe.

### Why allow large concurrent inner evaluation now?

Because the default configuration is no longer targeting a single local GPU. It is targeting rented multi-GPU execution where one rollout can be assigned to one GPU, which restores fair rollout timing and keeps the paper-like grouped rollout structure.

## Plain AutoResearch Mode Still Works

This fork does not remove the original autoresearch workflow. You can still use it directly:

```bash
uv run prepare.py
uv run train.py
```

The TTT-Discover path is an additional outer loop, not a replacement for the inner codebase.

## Current Readiness

What is tested locally:

- config loading and override behavior
- reward mapping
- candidate parsing
- environment prompt and state flow
- CLI wiring into upstream `discover`
- concurrency gating for inner evaluations

What is still environment-dependent:

- a true end-to-end production run on the target Linux/CUDA machine
- provider-specific model serving details
- real-world throughput and stability under long TTT sessions

So the repo is structurally ready for the intended setup, but final operational confidence still comes from a real GPU run on the target hardware.

## License

MIT
