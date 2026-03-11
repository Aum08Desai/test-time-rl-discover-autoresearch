from __future__ import annotations

import unittest

from ttt_autoresearch.prompt_builder import build_rollout_prompt


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_is_single_rollout_specific(self) -> None:
        prompt = build_rollout_prompt(
            state_ctx="You are iteratively optimizing val_bpb.\nCurrent val_bpb (lower is better): 1.020000",
            construction_section=(
                "You may want to start your search from the current training script shown above.\n"
                "This is the current starting point selected by the search procedure.\n"
                "Preserve a working script, but do not limit yourself to tiny hyperparameter tweaks.\n"
                "Pursue bold, high-upside changes when they are technically coherent and likely to materially improve val_bpb.\n"
                "You are encouraged to explore meaningfully different directions if the current approach appears saturated."
            ),
            code_section=(
                "Reason about how you could further improve this training script under the fixed 5-minute training budget.\n"
                "Hyperparameter tuning is allowed, but do not stop there: pursue stronger algorithmic, architectural, data-flow, attention, optimization, or systems ideas when they could deliver a step-change improvement.\n"
                "Prefer edits that are technically coherent and high-upside, even if they are more ambitious than simple hill-climbing.\n"
                "Try different algorithmic ideas, architecture changes, optimizer and schedule changes, batching changes, or other training heuristics.\n"
                "Moderate increases in VRAM are acceptable if they lead to meaningful gains.\n"
                "Do not refactor unrelated code, but do make all integration edits required for the new idea to work cleanly.\n"
                "Unless you make a meaningful improvement in `val_bpb`, you will not be rewarded."
            ),
        )
        self.assertIn("expert machine learning researcher", prompt)
        self.assertIn("## Problem", prompt)
        self.assertIn("## Budget & Resources", prompt)
        self.assertIn("## AutoResearch Invariants", prompt)
        self.assertIn("## Rules", prompt)
        self.assertIn("You are iteratively optimizing val_bpb.", prompt)
        self.assertIn("You may want to start your search from the current training script shown above.", prompt)
        self.assertIn("This is the current starting point selected by the search procedure.", prompt)
        self.assertIn("do not limit yourself to tiny hyperparameter tweaks", prompt)
        self.assertIn("Pursue bold, high-upside changes", prompt)
        self.assertIn("Reason about how you could further improve this training script under the fixed 5-minute training budget.", prompt)
        self.assertIn("Hyperparameter tuning is allowed, but do not stop there", prompt)
        self.assertIn("technically coherent and high-upside", prompt)
        self.assertIn("Moderate increases in VRAM are acceptable if they lead to meaningful gains.", prompt)
        self.assertIn("do make all integration edits required", prompt)
        self.assertIn("Maximum sequence length is `2048`", prompt)
        self.assertIn("Validation uses the pinned shard `06542`", prompt)
        self.assertIn("vocab size `8192`", prompt)
        self.assertIn("forward(x, y, reduction='none')", prompt)
        self.assertIn("TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0", prompt)
        self.assertIn("Preserve the final summary prints", prompt)
        self.assertIn("Return only one or more exact SEARCH/REPLACE edit blocks for `train.py`", prompt)
        self.assertIn("Prefer as few patch blocks as needed", prompt)
        self.assertIn("Treat each SEARCH block like an exact `old_string` tool argument", prompt)
        self.assertIn("Do not return standalone code fragments", prompt)
        self.assertIn("Do not wrap the answer in JSON", prompt)
        self.assertIn("Do not wrap the answer in markdown code fences", prompt)
        self.assertIn("Do not abbreviate with `...` or placeholders", prompt)
        self.assertIn("## Example Response", prompt)
        self.assertIn("<<<<<<< SEARCH", prompt)
        self.assertIn(">>>>>>> REPLACE", prompt)
        self.assertNotIn("Baseline val_bpb from the original script", prompt)
        self.assertNotIn("LOOP FOREVER", prompt)
        self.assertNotIn("results.tsv", prompt)
        self.assertNotIn("git reset", prompt)
        self.assertNotIn("NEVER STOP", prompt)


if __name__ == "__main__":
    unittest.main()
