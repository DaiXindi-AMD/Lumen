#!/usr/bin/env python3
"""CPU-only tests for the projection-guard campaign analyzer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))

import analyze_projection_guard as analysis  # noqa: E402


class ProjectionGuardAnalyzerTest(unittest.TestCase):
    def test_projection_argument_parser_is_exact(self) -> None:
        self.assertEqual(analysis.policy_projection_args(["train"]), ())
        self.assertEqual(
            analysis.policy_projection_args(
                [
                    "train",
                    "--mxfp4-last-layer-bf16-projections",
                    "o_proj",
                    "down_proj",
                ]
            ),
            ("o_proj", "down_proj"),
        )
        with self.assertRaises(ValueError):
            analysis.policy_projection_args(
                [
                    "train",
                    "--mxfp4-last-layer-bf16-projections",
                    "down_proj",
                    "--mxfp4-last-layer-bf16-projections",
                    "o_proj",
                ]
            )

    def test_policy_route_inventory_is_preregistered(self) -> None:
        self.assertEqual(len(analysis.EXPECTED_UNQUANTIZED["a"]), 8)
        self.assertEqual(len(analysis.EXPECTED_UNQUANTIZED["b"]), 3)
        self.assertEqual(len(analysis.EXPECTED_UNQUANTIZED["c"]), 2)
        self.assertEqual(analysis.EXPECTED_UNQUANTIZED["a"][-1], "lm_head")
        self.assertEqual(analysis.EXPECTED_UNQUANTIZED["b"][-1], "lm_head")
        self.assertEqual(analysis.EXPECTED_UNQUANTIZED["c"][-1], "lm_head")
        self.assertEqual(analysis.EXPECTED_ENABLED, {"a": 35, "b": 36, "c": 36})

    def test_policy_shape_counts_are_whole_smoke_totals(self) -> None:
        shapes = (
            (4096, 4096, 16384),
            (4096, 12288, 16384),
            (6144, 4096, 16384),
            (12288, 4096, 16384),
            (16384, 4096, 4096),
            (16384, 4096, 6144),
            (16384, 4096, 12288),
            (16384, 6144, 4096),
            (16384, 12288, 4096),
        )
        expected = {
            "a": (840, 840, 840, 1680, 2240, 840, 3080, 1400, 3640),
            "b": (840, 840, 864, 1728, 2240, 864, 3128, 1440, 3720),
            "c": (864, 840, 864, 1728, 2304, 864, 3128, 1440, 3720),
        }
        for policy, totals in expected.items():
            self.assertEqual(
                tuple(
                    analysis.EXPECTED_POLICY_SHAPE_TOTALS[policy][shape]
                    for shape in shapes
                ),
                totals,
            )

    def test_all_eight_policy_b_fixture_ranks_match_total_counts(self) -> None:
        self.assertEqual(
            analysis.fixture_bundle_sha256(), analysis.EXPECTED_FIXTURE_SHA
        )
        ok, errors = analysis.shape_inventory(analysis.FIXTURE_DIR, "b")
        self.assertTrue(ok, errors)
        self.assertFalse(analysis.shape_inventory(analysis.FIXTURE_DIR, "a")[0])
        self.assertFalse(analysis.shape_inventory(analysis.FIXTURE_DIR, "c")[0])

    def test_train_hash_pairing_is_formal_and_equal_cardinality_only(self) -> None:
        def case(name: str, train_samples: int, marker: str) -> dict:
            return {
                "name": name,
                "meta": {"command": f"train --train-samples {train_samples}"},
                "train_digests": {rank: f"{marker}-{rank}" for rank in range(8)},
            }

        cases = {
            analysis.SMOKE: case(analysis.SMOKE, 6400, "smoke"),
            "tail1_a1": case("tail1_a1", 6400, "formal-reference"),
            "guard_o_down_b1": case("guard_o_down_b1", 6400, "formal-candidate"),
            "guard_down_c": case("guard_down_c", 128, "different-cardinality"),
        }
        references = analysis.formal_train_digest_references(cases)
        self.assertEqual(set(references), {"tail1_a1", "guard_o_down_b1"})
        self.assertEqual(references["tail1_a1"], cases["tail1_a1"]["train_digests"])
        self.assertEqual(
            references["guard_o_down_b1"], cases["tail1_a1"]["train_digests"]
        )

    def test_bootstrap_is_paired_and_deterministic(self) -> None:
        control = [100.0 + index / 100 for index in range(40)]
        candidate = [98.0 + index / 100 for index in range(40)]
        first = analysis.base.bootstrap(control, candidate, resamples=300)
        second = analysis.base.bootstrap(control, candidate, resamples=300)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_points"], 40)
        self.assertEqual(first["block_length"], 4)
        self.assertGreater(first["speedup_ci95"][0], 1.0)

    def test_speed_gate_boundaries(self) -> None:
        self.assertTrue(
            analysis.base.performance_pass(1.003, 1.003, 28, [1.000001, 1.02])
        )
        self.assertFalse(
            analysis.base.performance_pass(1.003, 1.003, 27, [1.000001, 1.02])
        )
        self.assertFalse(analysis.base.performance_pass(1.003, 1.003, 40, [1.0, 1.02]))

    def test_selection_requires_complete_chain(self) -> None:
        all_pass = {"a_to_b": True, "b_to_c": True, "a_to_c": True}
        selected, chain = analysis.select_policy(
            integrity=True,
            drift=True,
            performance=all_pass,
            precision_b=True,
            precision_c=True,
        )
        self.assertEqual(selected, "C")
        self.assertTrue(chain["c_complete_chain_pass"])

        selected, chain = analysis.select_policy(
            integrity=True,
            drift=True,
            performance={**all_pass, "b_to_c": False},
            precision_b=True,
            precision_c=True,
        )
        self.assertEqual(selected, "B")
        self.assertFalse(chain["c_complete_chain_pass"])

        selected, _chain = analysis.select_policy(
            integrity=True,
            drift=True,
            performance={**all_pass, "a_to_b": False},
            precision_b=True,
            precision_c=True,
        )
        self.assertEqual(selected, "A")

        selected, _chain = analysis.select_policy(
            integrity=False,
            drift=True,
            performance=all_pass,
            precision_b=True,
            precision_c=True,
        )
        self.assertIsNone(selected)

    def test_formal_policy_commands(self) -> None:
        common = (
            "/usr/local/bin/torchrun --nproc-per-node 8 "
            f"{analysis.ENTRY} --seq-length 8192 --micro-batch-size 2 "
            "--gradient-accumulation-steps 8 --max-steps 50 "
            "--train-samples 6400 --eval-interval 50 --eval-batches 16 "
            "--val-samples 256 --seed 1234 --mode mxfp4 --fsdp-version 2 "
            "--sharding full_shard --fsdp-reduce-dtype bf16 "
            "--num-layers-at-start-in-bf16 0 "
        )
        cases = {
            "tail1_a1": common + "--num-layers-at-end-in-bf16 1",
            "guard_o_down_b1": common
            + "--num-layers-at-end-in-bf16 0 "
            + "--mxfp4-last-layer-bf16-projections o_proj down_proj",
            "guard_down_c": common
            + "--num-layers-at-end-in-bf16 0 "
            + "--mxfp4-last-layer-bf16-projections down_proj",
        }
        for name, command in cases.items():
            self.assertTrue(
                analysis.policy_command_check(
                    {"name": name, "meta": {"command": command}}
                )
            )
        normalized = [
            analysis.normalized_policy_command(value) for value in cases.values()
        ]
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])


if __name__ == "__main__":
    unittest.main()
