#!/usr/bin/env python3
"""CPU/static tests for the fresh policy_a profiling harness."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_policy_a_profile.sh"
ROUTE = ROOT / "route_entry.py"
ANALYZER = ROOT / "analyze_policy_a_profile.py"
ANALYZER_TEST = ROOT / "test_analyze_policy_a_profile.py"
PROTOCOL = ROOT / "protocol.md"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("policy_a_profile_analyzer", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load analyzer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestHarnessSyntax(unittest.TestCase):
    def test_python_files_parse_without_importing_gpu_modules(self):
        for path in (ROUTE, ANALYZER, ANALYZER_TEST, Path(__file__)):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_shell_parses_from_unrelated_working_directory(self):
        result = subprocess.run(
            ["/usr/bin/bash", "-n", str(RUNNER)],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runner_resolves_its_own_root(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('dirname -- "${BASH_SOURCE[0]}"', source)
        self.assertIn("-m unittest discover", source)
        self.assertIn('-s "${ROOT}"', source)


class TestCampaignContract(unittest.TestCase):
    def setUp(self):
        self.runner = RUNNER.read_text(encoding="utf-8")
        self.protocol = PROTOCOL.read_text(encoding="utf-8")

    def test_exact_arm_names_and_order_are_present(self):
        expected = (
            "smoke_mxfp4_policy_a",
            "replay_mxfp4_policy_a",
            "profile_bf16",
            "profile_mxfp4_policy_a",
        )
        invocations = (
            "run_case smoke_mxfp4_policy_a",
            "run_case replay_mxfp4_policy_a",
            "run_case profile_bf16",
            "run_case profile_mxfp4_policy_a",
        )
        positions = [self.runner.index(invocation) for invocation in invocations]
        self.assertEqual(positions, sorted(positions))
        for name in expected:
            self.assertIn(name, self.protocol)

    def test_profile_window_and_tail_contract(self):
        self.assertIn("LUMEN_PROF_START=7", self.runner)
        self.assertIn("LUMEN_PROF_END=8", self.runner)
        self.assertIn("--num-layers-at-end-in-bf16 1", self.runner)
        self.assertIn("expected_quantized_linears=245", self.runner)
        self.assertIn("expected_bf16_skipped_linears=8", self.runner)
        self.assertIn("expected_packed_qkv_layers=35", self.runner)
        self.assertIn("expected_split_swiglu_layers=35", self.runner)
        self.assertIn("readonly TRAIN_SAMPLES_FORMAL=1024", self.runner)
        self.assertIn("TRAIN_SAMPLES_FORMAL == FORMAL_STEPS * 128", self.runner)

    def test_route_success_counts_are_exact(self):
        self.assertIn('"qkv_linear_success": 1400', self.runner)
        self.assertIn('"swiglu_fwd_success": 1400', self.runner)
        self.assertIn('"swiglu_bwd_success": 840', self.runner)

    def test_fresh_cache_then_replay_then_independent_copies(self):
        self.assertIn("CACHE_BUILD_DIR", self.runner)
        self.assertIn("require_fresh_cache", self.runner)
        self.assertIn("validate_replay_identity", self.runner)
        self.assertIn("BF16_CACHE_DIR", self.runner)
        self.assertIn("MXFP4_CACHE_DIR", self.runner)
        self.assertIn("cp --reflink=auto", self.runner)

    def test_each_arm_rechecks_workload_and_live_asm_identity(self):
        self.assertIn(
            'verify_current_state 1 || fail "${label}: frozen source/workload state changed before launch"',
            self.runner,
        )
        self.assertIn("if ! verify_current_state 1; then", self.runner)
        for field in (
            "manifest_sha256",
            "code_object",
            "code_object_sha256",
            "split_k_capable",
        ):
            self.assertIn(field, self.runner)
        self.assertIn("ASM live artifact identity mismatch", self.runner)
        self.assertGreaterEqual(self.runner.count(".hexdigest()[:16]"), 2)

    def test_campaign_root_rejects_unknown_preexisting_entries(self):
        self.assertIn("unexpected pre-existing campaign-root entry", self.runner)
        self.assertIn('find "${ROOT}" -mindepth 1 -maxdepth 1', self.runner)

    def test_no_old_profile_result_is_an_input(self):
        forbidden = (
            "tail0-bf16-confirm",
            "policy_a10-confirm",
            "tail-boundary-fresh",
            "packed-qkv-profile-fresh",
            "post-packed-profile",
            "tail4-profile",
            "UNION_CACHE",
            "OLD_TRACE_PARSER",
        )
        for token in forbidden:
            self.assertNotIn(token, self.runner)

    def test_required_status_and_hash_artifacts(self):
        required = (
            "profile-meta.txt",
            "profile-complete.txt",
            "policy_a-profile-exit-status.txt",
            "source-audit.json",
            "campaign-artifacts.sha256",
            "test_analyze_policy_a_profile.py",
            "train-exit-status.txt",
            "postflight-status.txt",
            "source-bundle-before.sha256",
            "source-bundle-after.sha256",
            "tree-state-after.txt",
        )
        for name in required:
            self.assertIn(name, self.runner)
        self.assertIn("analyzer_test_sha256", self.runner)

    def test_kfd_and_lock_are_fail_closed(self):
        self.assertIn("/dev/kfd must be readable and writable", self.runner)
        self.assertIn(".lumen-gpu-exclusive.lock", self.runner)
        self.assertIn("non_service_kfd_client", self.runner)
        self.assertIn("flock -n", self.runner)
        self.assertIn('comm=$(cat -- "/proc/${pid}/comm"', self.runner)

    def test_python_bytecode_is_disabled_and_rejected(self):
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", self.runner)
        self.assertIn("/usr/bin/python3 -B", self.runner)
        self.assertIn("Python bytecode artifacts must be absent", self.runner)
        self.assertIn("! -path '*/__pycache__/*'", self.runner)

    def test_finalization_is_hashed_then_verified_without_rewriting_analysis(self):
        progress = self.runner.index("stage=manifest_ready")
        manifest = self.runner.index("manifest_temporary=${MANIFEST}.tmp.$$")
        complete = self.runner.index('write_atomic "${COMPLETE}"')
        final_verify = self.runner.index("--verify-completion")
        self.assertLess(progress, manifest)
        self.assertLess(manifest, complete)
        self.assertLess(complete, final_verify)
        self.assertIn('"profile_meta_sha256=$(sha_file "${META}")"', self.runner)
        self.assertIn(
            '"campaign_stage_status_sha256=$(sha_file "${STATUS_FILE}")"',
            self.runner,
        )

    def test_route_flush_is_explicit_and_finally_guarded(self):
        tree = ast.parse(ROUTE.read_text(encoding="utf-8"))
        finally_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        self.assertTrue(any(node.finalbody for node in finally_nodes))
        self.assertIn("_save_shape_log()", ROUTE.read_text(encoding="utf-8"))


class TestAnalyzerHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analyzer()

    def test_interval_union_is_overlap_safe(self):
        self.assertEqual(self.analysis.interval_union([]), 0.0)
        self.assertEqual(
            self.analysis.interval_union([(0.0, 2.0), (1.0, 3.0), (5.0, 7.0)]),
            5.0,
        )

    def test_command_normalization_removes_only_allowed_deltas(self):
        bf16 = "torchrun train.py --mode bf16 --seed 1234"
        policy_a = (
            "torchrun train.py --mode mxfp4 --seed 1234 "
            "--mxfp4-pack-qkv --mxfp4-fuse-swiglu"
        )
        self.assertEqual(
            self.analysis.normalize_command(bf16),
            self.analysis.normalize_command(policy_a),
        )
        changed = "torchrun train.py --mode mxfp4 --seed 999 --mxfp4-pack-qkv"
        self.assertNotEqual(
            self.analysis.normalize_command(bf16),
            self.analysis.normalize_command(changed),
        )

    def test_classifier_keeps_unclassified_work(self):
        event = self.analysis.GPUEvent(
            0.0, 1.0, "kernel", "unknown_kernel", None, None, 0
        )
        self.assertEqual(self.analysis.classify_event(event), "other")

    def test_analyzer_does_not_use_bad_adjacent_zip(self):
        source = ANALYZER.read_text(encoding="utf-8")
        self.assertNotIn("zip(ordered, ordered[1:], strict=True)", source)
        self.assertNotIn("packed_dw_contiguous > 0", source)


if __name__ == "__main__":
    unittest.main()
