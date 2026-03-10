# Test Time RL Discover + Auto Research

![teaser](progress.png)

This repo is a focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that replaces the outer experiment loop with [TTT-Discover](https://github.com/test-time-training/discover).

The checked-in default is now a practical unattended setup:

- **Outer loop:** Tinker + `openai/gpt-oss-120b`
- **Renderer:** `gpt_oss_high_reasoning`
- **Inner loop:** RunPod `H100 PCIe` spot workers
- **Main preset:** `2 groups x 8 rollouts x 12 steps`
- **Spot failover:** if a worker pod is preempted, the current rollout is retried on a replacement pod automatically

The core objective stays the same as the original AutoResearch repo: improve [`train.py`](train.py) to lower `val_bpb`.

## Credits

This project builds on:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- [test-time-training/discover](https://github.com/test-time-training/discover)

The RL recipe stays with upstream `discover`. This repo provides the AutoResearch-specific environment, reward, runner, RunPod execution backend, and practical launch workflow.

## How The System Works

There are two loops:

1. **Inner loop**
   - [`train.py`](train.py) is the only file the outer model edits.
   - Every rollout runs a real fixed-budget AutoResearch training job.
   - The score is `val_bpb`, and lower is better.

2. **Outer loop**
   - TTT-Discover samples full-file replacements for `train.py`.
   - Each candidate is evaluated by the inner loop.
   - Reward is a direct transformed task score: `1 / (1e-8 + val_bpb)`.
   - Failed or invalid candidates receive `0.0` reward.
   - Upstream `discover` updates the outer model online.

The checked-in workflow keeps the outer controller on a stable machine and uses RunPod spot instances only for the inner evaluations. That is what lets the run continue unattended if a spot worker disappears.

## What “Unattended” Means Here

This repo is designed so that:

- the controller process running [`run_ttt_discover.py`](run_ttt_discover.py) stays alive on a stable machine
- inner evaluations are dispatched to RunPod spot pods
- if a pod is preempted during a rollout, the runner provisions a replacement pod
- the interrupted rollout is retried from scratch on the replacement pod
- the run continues until the configured `groups_per_step x samples_per_step x max_steps` budget is completed

Important boundary:

- the controller process itself is **not** spot-resilient
- only the **inner worker pool** is spot-resilient

So the outer process should run on your laptop, workstation, or another non-preemptible box. The H100 spot instances are only for the expensive inner `train.py` jobs.

## Current Default

The default config at [`configs/ttt_discover_autoresearch.yaml`](configs/ttt_discover_autoresearch.yaml) is the recommended medium run:

- `model_name: openai/gpt-oss-120b`
- `renderer_name: gpt_oss_high_reasoning`
- `target_val_bpb: 0.85`
- `execution_backend: runpod`
- `groups_per_step: 2`
- `samples_per_step: 8`
- `max_steps: 12`
- `max_concurrent_evaluations: 16`
- `runpod_gpu_type_ids: ["NVIDIA H100 PCIe"]`
- `runpod_interruptible: true`

That means:

- `16` rollouts per outer step
- `12` outer RL updates
- `192` rollout evaluations total
- `1` extra baseline run before RL starts
- `193` total inner jobs

## Presets

The repo ships with three practical presets:

### Small

File: [`configs/ttt_discover_autoresearch_small.yaml`](configs/ttt_discover_autoresearch_small.yaml)

- `2 x 4 x 12`
- `96` RL rollouts
- `8` concurrent RunPod workers

### Medium

File: [`configs/ttt_discover_autoresearch_medium.yaml`](configs/ttt_discover_autoresearch_medium.yaml)

- `2 x 8 x 12`
- `192` RL rollouts
- `16` concurrent RunPod workers

This is the recommended main mode and matches the default config.

### Large

File: [`configs/ttt_discover_autoresearch_large.yaml`](configs/ttt_discover_autoresearch_large.yaml)

- `2 x 8 x 20`
- `320` RL rollouts
- `16` concurrent RunPod workers

Use this only after the medium run is stable.

## RunPod Backend

The inner-loop executor now supports two backends:

- `local`
- `runpod`

The `runpod` backend does the following:

1. Creates up to `max_concurrent_evaluations` spot pods.
2. Waits for SSH on each pod.
3. Bootstraps the pod by:
   - uploading the repo snapshot
   - installing `uv`
   - running `uv sync`
   - running `uv run prepare.py --num-shards 10`
4. Uploads each candidate workspace to a worker pod.
5. Runs the inner command remotely.
6. Pulls back `stdout.log`, `stderr.log`, and metrics.
7. Deletes the pods automatically when the run finishes.

If a pod disappears during upload, bootstrap, or execution, the worker is retired, a replacement is created, and the interrupted rollout is retried.

## Prerequisites

You need:

- Linux or macOS for the controller machine
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- a Tinker-enabled account for the outer loop
- a RunPod account with:
  - API access
  - an SSH public key registered in the account
  - access to H100 spot instances

Environment:

```bash
export RUNPOD_API_KEY=...
```

You also need whatever Tinker credentials your local `ttt-discover` installation expects.

## Quick Start

Launch the default unattended medium run:

```bash
uv sync
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

Or explicitly choose the medium preset:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch_medium.yaml
```

## Cost And Runtime Shape

For this repo, the expensive part is the inner loop. Each rollout is a real five-minute AutoResearch training job.

The repo’s own reference timing in [`program.md`](program.md) shows:

- `total_seconds: 325.9` per rollout

That means the default medium run has:

- `192` RL rollouts
- `1` baseline run
- `193` total inner runs
- about `17.47` total GPU-hours

### Example Medium Budget

Using your current spot numbers:

- `H100 PCIe spot: $1.25/hr`
- `H100 SXM spot: $1.75/hr`

The medium run works out to:

- `H100 PCIe`: about `$21.84`
- `H100 SXM`: about `$30.57`

Tinker is the smaller cost bucket here. The exact amount depends on current Tinker pricing and token usage, but for this repo it is materially smaller than the H100 rental line item.

### Wall Clock

Total GPU-hours are roughly fixed, so more pods mostly reduce elapsed time, not total spend.

Approximate medium-run wall clock:

- `1 H100`: about `18-20h`
- `8 H100s`: about `2.5-3h`
- `16 H100s`: about `1.3-1.8h`

## Model And Renderer

The checked-in default is:

```yaml
model_name: openai/gpt-oss-120b
renderer_name: gpt_oss_high_reasoning
```

This is intentional:

- it matches the strongest paper-aligned model family more closely than the older Qwen default
- it is already supported by the renderer mapping in [`ttt_autoresearch/config.py`](ttt_autoresearch/config.py)
- it is the intended outer-loop model for the default RunPod workflow

Other models still work, but if the model family is not recognized automatically you must set `renderer_name` explicitly.

## Important Config Knobs

The main knobs for unattended RunPod execution are:

- `execution_backend`
  - use `runpod` for remote spot workers
  - use `local` for direct local GPU execution
- `max_concurrent_evaluations`
  - number of worker pods for `runpod`
  - number of local simultaneous inner runs for `local`
- `runpod_gpu_type_ids`
  - default is `["NVIDIA H100 PCIe"]`
- `runpod_interruptible`
  - leave this `true` for spot behavior
- `runpod_bootstrap_commands`
  - optional override if you want to use a custom image or template
- `runpod_retry_limit`
  - how many times to reprovision and retry an interrupted rollout before surfacing a failure

## Fixed Prompt Target

The checked-in presets use:

```yaml
target_val_bpb: 0.85
```

This is a prompt-side benchmark target, not a reward cap.

- the model is shown the current starting state and the gap to `0.85`
- the RL reward still comes from the actual achieved `val_bpb`
- if a rollout beats `0.85`, it is still rewarded more for going even lower

This mirrors how upstream `discover` environments use fixed benchmark targets in the prompt while computing reward from the evaluated task score.

## Repository Layout

```text
prepare.py                  Fixed data prep and runtime utilities
train.py                    Inner training program edited by the outer model
program.md                  Human-authored research instructions/context
run_ttt_discover.py         Main TTT-Discover entrypoint
ttt_autoresearch/           Adapter layer for environment, reward, runner, RunPod, config
configs/                    Practical preset YAML configs
tests/                      Smoke and unit coverage for the adapter
```

## Output Artifacts

Each run writes artifacts under `runs/<timestamp>/`:

- `baseline.json`
- `resolved_config.json`
- `history.jsonl`
- `best/train.py`
- `best/metrics.json`
- `candidates/`
- `discover_log/`
- `runpod_pool.json`

`runpod_pool.json` records the worker pod metadata for the current run so you can inspect what was provisioned.

The important resume/checkpoint files are:

- `baseline.json`
  - cached baseline result; if it already exists, the CLI reuses it instead of rerunning baseline
- `baseline/train.py`
  - stored baseline script snapshot for reproducible resume
- `best/train.py`
  - best discovered script so far
- `best/metrics.json`
  - best discovered `val_bpb` plus artifact paths
- `history.jsonl`
  - append-only candidate evaluation log
- `candidates/<step>_<id>/train.py`
  - exact candidate script evaluated for that rollout
- `candidates/<step>_<id>/stdout.log`
  - raw stdout from the inner AutoResearch run
- `candidates/<step>_<id>/stderr.log`
  - raw stderr from the inner AutoResearch run
- `candidates/<step>_<id>/metrics.json`
  - parsed metrics sidecar for that rollout
- `candidates/<step>_<id>/rollout_manifest.json`
  - self-contained rollout record with the starting state, candidate payload, evaluation result, reward, and promotion outcome
- invalid or malformed model outputs are also persisted under `candidates/` with a `rollout_manifest.json`, `metrics.json`, and raw `response.txt`
- `discover_log/checkpoints.jsonl`
  - upstream TTT-Discover checkpoint index
- `discover_log/`
  - LoRA/training state and sampler checkpoints used for resume

## Resuming A Stopped Run

To continue a stopped run, reuse the same `run_dir`.

Example:

```yaml
run_dir: runs/my-main-run
```

Then rerun the same command:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

Resume behavior:

- the CLI reuses `baseline.json` and `baseline/train.py` if they already exist
- upstream `discover` reloads the latest training checkpoint from `discover_log/checkpoints.jsonl`
- upstream sampler state is reloaded from the matching sampler checkpoint step
- every evaluated rollout remains on disk under `candidates/`, so prompt/response/result provenance is preserved even if the run is interrupted

If you stopped at `12` steps and want to continue farther, increase `max_steps` above the completed count before rerunning.

For example, to continue a finished medium run out to `20` steps:

- keep the same `run_dir`
- change `max_steps: 20`
- rerun the command

Important resume rule:

- resume with the same code revision, model, renderer, rollout structure, and run directory whenever possible
- changing those mid-run is not guaranteed to be meaningful or stable

## Local Mode Still Exists

If you want to run without RunPod, set:

```yaml
execution_backend: local
```

and configure `gpu_devices` if you want more than one local concurrent evaluation.

## Current Readiness

What is covered in tests:

- config loading and normalization
- reward mapping
- candidate parsing
- CLI wiring into upstream `discover`
- local concurrency gating
- RunPod retry logic for interrupted workers
- runner cleanup behavior

What is still operationally environment-dependent:

- real RunPod API credentials
- SSH access from the controller to the worker pods
- real Tinker credentials and provider setup
- long-run stability on your specific account and spot market

So the repo is structurally ready for unattended Tinker + RunPod operation, but the final production proof is still a real run on your account.

## License

MIT
