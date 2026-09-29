#!/usr/bin/env python3
"""CPU-only contract tests for the final Policy A BF16 bracket analyzer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import analyze_final_policy_a_bf16_bracket as analysis


def command(precision: str, entry: Path, steps: int) -> str:
    tokens = [
        "/usr/local/bin/torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(entry),
        "--init-from-scratch",
        "--seq-length",
        "8192",
        "--micro-batch-size",
        "2",
        "--gradient-accumulation-steps",
        "8",
        "--max-steps",
        str(steps),
        "--train-samples",
        "6400",
        "--eval-interval",
        str(steps),
        "--eval-batches",
        "16",
        "--val-samples",
        "256",
        "--seed",
        "1234",
        "--mode",
        precision,
        "--fsdp-version",
        "2",
        "--sharding",
        "full_shard",
        "--fsdp-reduce-dtype",
        "bf16",
        "--first-last-layers-bf16",
        "--num-layers-at-start-in-bf16",
        "0",
        "--num-layers-at-end-in-bf16",
        "1",
        "--fsdp-retain-accumulated-params",
        "--no-grad-checkpointing",
    ]
    if precision == "mxfp4":
        tokens.extend(("--mxfp4-pack-qkv", "--mxfp4-fuse-swiglu"))
    return " ".join(tokens)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = {"global_batch": "128", "gradient_accumulation": "8"}

    def test_registered_order_and_gates(self) -> None:
        self.assertEqual(
            analysis.ORDER,
            ("smoke_tail1", "bf16_a1", "mxfp4_policy_a", "bf16_a2"),
        )
        self.assertEqual(analysis.WINDOW, tuple(range(11, 51)))
        self.assertEqual(analysis.MIN_SPEEDUP, 1.6)
        self.assertEqual(analysis.MIN_WINS, 28)
        self.assertEqual(analysis.BLOCK_LENGTH, 4)
        self.assertEqual(analysis.BOOTSTRAP_RESAMPLES, 100_000)

    def test_precision_commands_and_normalization(self) -> None:
        bf16 = command("bf16", analysis.TRAIN_ENTRY, 50)
        candidate = command("mxfp4", analysis.TRAIN_ENTRY, 50)
        smoke = command("mxfp4", analysis.ENTRY, 3)
        self.assertTrue(
            all(
                analysis.command_checks(
                    bf16,
                    precision="bf16",
                    total_steps=50,
                    meta=self.meta,
                    smoke=False,
                ).values()
            )
        )
        self.assertTrue(
            all(
                analysis.command_checks(
                    candidate,
                    precision="mxfp4",
                    total_steps=50,
                    meta=self.meta,
                    smoke=False,
                ).values()
            )
        )
        self.assertTrue(
            all(
                analysis.command_checks(
                    smoke,
                    precision="mxfp4",
                    total_steps=3,
                    meta=self.meta,
                    smoke=True,
                ).values()
            )
        )
        self.assertEqual(
            analysis.normalized_formal_command(bf16),
            analysis.normalized_formal_command(candidate),
        )
        bad_bf16 = bf16 + " --mxfp4-pack-qkv"
        checks = analysis.command_checks(
            bad_bf16,
            precision="bf16",
            total_steps=50,
            meta=self.meta,
            smoke=False,
        )
        self.assertFalse(checks["precision_flags"])
        bad_candidate = candidate + " --mxfp4-last-layer-bf16-projections o_proj"
        checks = analysis.command_checks(
            bad_candidate,
            precision="mxfp4",
            total_steps=50,
            meta=self.meta,
            smoke=False,
        )
        self.assertFalse(checks["no_projection_override"])

    def test_performance_and_strict_drift_boundaries(self) -> None:
        self.assertTrue(analysis.performance_pass(1.6, 1.6, 28, [1.6, 1.7]))
        self.assertFalse(
            analysis.performance_pass(1.6, 1.6, 28, [1.599999, 1.7])
        )
        cases = {
            "bf16_a1": {"window": [100.0] * 40},
            "bf16_a2": {"window": [102.999] * 40},
        }
        self.assertTrue(analysis.bf16_drift(cases)["pass"])
        cases["bf16_a2"]["window"] = [103.0] * 40
        self.assertFalse(analysis.bf16_drift(cases)["pass"])

    def test_exact_cache_identity(self) -> None:
        profiles = {}
        for shape, identity in analysis.EXPECTED_ASM_IDENTITIES.items():
            profiles[shape] = {
                "winner": "asm",
                "incumbent": "asm",
                "decision_scope": "single_device",
                "profile_device_count": 1,
                "selection_policy": "protected_asm_consensus_required",
                "reason": analysis.EXPECTED_ASM_REASON,
                "switch_margin": 1.05,
                "identities": {"asm": {"implementation": "asm", **identity}},
            }
        payload = {
            "schema": 6,
            "arch": "gfx950",
            "backends": analysis.EXPECTED_CACHE_BACKENDS,
            "tuned_tables": analysis.EXPECTED_TUNED_TABLES,
            "decision_scope": "single_device",
            "profile_device_count": 1,
            "profile_settings": {"timed_iters": 11, "warmup_iters": 3},
            "choices": analysis.EXPECTED_CACHE_CHOICES,
            "profiles": profiles,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "cache.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(analysis.cache_integrity(path)[0])
            payload["profiles"]["6144,4096,16384"]["identities"]["asm"][
                "code_object_sha256"
            ] = "bad"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(analysis.cache_integrity(path)[0])

    def test_policy_a_route_and_shape_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            for rank in range(8):
                report = {
                    "rank": rank,
                    "counts": {
                        "rank": rank,
                        "world_size": 8,
                        "inventory_captured": 1,
                        "enabled_qkv": 35,
                        "enabled_swiglu": 35,
                    },
                    "unquantized_linear_names": analysis.EXPECTED_UNQUANTIZED,
                    "lm_head_count": 1,
                    "lm_head_weight_dtype": "bfloat16",
                    "lm_head_quant_enabled": False,
                    "lumen_import": "/home/xdai/Lumen/lumen/__init__.py",
                    "aiter_import": "/home/xdai/aiter/aiter/__init__.py",
                }
                (directory / f"route-rank{rank}.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                lines = ["M,N,K,asm_available,backend,calls"]
                lines.extend(
                    f"{m},{n},{k},1,asm,{calls}"
                    for (m, n, k), calls in analysis.EXPECTED_SMOKE_SHAPES.items()
                )
                (directory / f"mxfp4-shapes-rank{rank}.csv").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            passed, errors = analysis.smoke_route_integrity(directory)
            self.assertTrue(passed, errors)

    def test_manifest_is_exact_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            first = directory / "first"
            second = directory / "second"
            first.write_text("a\n", encoding="utf-8")
            second.write_text("b\n", encoding="utf-8")
            manifest = directory / "manifest"
            manifest.write_text(
                f"{analysis.base.sha256(first)}  {first}\n"
                f"{analysis.base.sha256(second)}  {second}\n",
                encoding="utf-8",
            )
            self.assertTrue(
                analysis.checksum_manifest_ok(manifest, {first, second})
            )
            second.write_text("changed\n", encoding="utf-8")
            self.assertFalse(
                analysis.checksum_manifest_ok(manifest, {first, second})
            )

    def test_phase1_boundary_rejects_bad_schema_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sentinel = directory / "phase1-complete.txt"
            manifest = directory / "phase1-artifacts.sha256"
            postflight = directory / "postflight-status.txt"
            aiter_cache = directory / "aiter-configs"
            aiter_cache.mkdir()
            manifest.write_text("manifest\n", encoding="utf-8")
            postflight.write_text("postflight\n", encoding="utf-8")
            postflight_ns = 1_700_000_000_000_000_000
            manifest_ns = postflight_ns + 1_000_000
            completed_ns = manifest_ns + 1_000_000
            pre_a2_ns = completed_ns + 1_000_000
            os.utime(postflight, ns=(postflight_ns, postflight_ns))
            os.utime(manifest, ns=(manifest_ns, manifest_ns))

            def checks(schema: str, timestamp: str) -> dict[str, bool]:
                sentinel.write_text(
                    f"schema={schema}\ncompleted_epoch_ns={timestamp}\n",
                    encoding="utf-8",
                )
                return analysis.phase1_boundary_checks(
                    phase1_complete=sentinel,
                    phase1_manifest=manifest,
                    candidate_postflight=postflight,
                    aiter_config_cache=aiter_cache,
                    pre_a2_epoch_ns=pre_a2_ns,
                )

            self.assertTrue(all(checks("1", str(completed_ns)).values()))
            self.assertFalse(checks("2", str(completed_ns))["phase1_sentinel_schema"])
            self.assertFalse(
                checks("1", "not-a-timestamp")["phase1_completed_epoch_ns_format"]
            )
            self.assertFalse(
                checks("1", str(completed_ns // 10))[
                    "phase1_completed_epoch_ns_format"
                ]
            )
            self.assertFalse(
                checks("1", str(postflight_ns - 1))[
                    "phase1_sentinel_after_candidate_postflight"
                ]
            )
            self.assertFalse(
                checks("1", str(manifest_ns - 1))["phase1_sentinel_after_manifest"]
            )
            self.assertFalse(
                checks("1", str(pre_a2_ns))[
                    "phase2_preflight_after_phase1_completion"
                ]
            )

    def test_phase1_boundary_requires_real_empty_aiter_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sentinel = directory / "phase1-complete.txt"
            manifest = directory / "phase1-artifacts.sha256"
            postflight = directory / "postflight-status.txt"
            aiter_cache = directory / "aiter-configs"
            completed_ns = 1_700_000_000_002_000_000
            sentinel.write_text(
                f"schema=1\ncompleted_epoch_ns={completed_ns}\n",
                encoding="utf-8",
            )
            manifest.write_text("manifest\n", encoding="utf-8")
            postflight.write_text("postflight\n", encoding="utf-8")
            os.utime(manifest, ns=(completed_ns - 1, completed_ns - 1))
            os.utime(postflight, ns=(completed_ns - 2, completed_ns - 2))

            def cache_check() -> bool:
                return analysis.phase1_boundary_checks(
                    phase1_complete=sentinel,
                    phase1_manifest=manifest,
                    candidate_postflight=postflight,
                    aiter_config_cache=aiter_cache,
                    pre_a2_epoch_ns=completed_ns + 1,
                )["aiter_config_cache_real_empty_directory"]

            self.assertFalse(cache_check())
            aiter_cache.mkdir()
            self.assertTrue(cache_check())
            (aiter_cache / "unexpected").mkdir()
            self.assertFalse(cache_check())
            (aiter_cache / "unexpected").rmdir()
            aiter_cache.rmdir()
            target = directory / "actual-aiter-configs"
            target.mkdir()
            aiter_cache.symlink_to(target, target_is_directory=True)
            self.assertFalse(cache_check())

    def test_phase2_wires_boundary_validation_before_bf16_a2(self) -> None:
        driver = analysis.DRIVER.read_text(encoding="utf-8")
        verify_start = driver.index("verify_phase1() {")
        verify_end = driver.index("\n}\n\nrun_phase1()", verify_start)
        verify_block = driver[verify_start:verify_end]
        self.assertIn("--validate-phase1-boundary", verify_block)
        self.assertLess(
            verify_block.index("verify_phase1_manifest"),
            verify_block.index("--validate-phase1-boundary"),
        )
        phase2_start = driver.index("run_phase2() {")
        phase2_end = driver.index("\n}\n\nif [[", phase2_start)
        phase2_block = driver[phase2_start:phase2_end]
        self.assertLess(
            phase2_block.index("verify_phase1"),
            phase2_block.index("run_arm bf16_a2"),
        )


if __name__ == "__main__":
    unittest.main()
