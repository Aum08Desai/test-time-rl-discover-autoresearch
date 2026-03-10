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
                "You are encouraged to explore meaningfully different directions if the current approach appears saturated."
            ),
            code_section=(
                "Reason about how you could further improve this training script under the fixed 5-minute training budget.\n"
                "Try different algorithmic ideas, architecture changes, optimizer and schedule changes, batching changes, or other training heuristics.\n"
                "Moderate increases in VRAM are acceptable if they lead to meaningful gains.\n"
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
        self.assertIn("Reason about how you could further improve this training script under the fixed 5-minute training budget.", prompt)
        self.assertIn("Moderate increases in VRAM are acceptable if they lead to meaningful gains.", prompt)
        self.assertIn("Maximum sequence length is `2048`", prompt)
        self.assertIn("Validation uses the pinned shard `06542`", prompt)
        self.assertIn("vocab size `8192`", prompt)
        self.assertIn("forward(x, y, reduction='none')", prompt)
        self.assertIn("Return exactly one ```json``` block", prompt)
        self.assertNotIn("Baseline val_bpb from the original script", prompt)
        self.assertNotIn("LOOP FOREVER", prompt)
        self.assertNotIn("results.tsv", prompt)
        self.assertNotIn("git reset", prompt)
        self.assertNotIn("NEVER STOP", prompt)


if __name__ == "__main__":
    unittest.main()
