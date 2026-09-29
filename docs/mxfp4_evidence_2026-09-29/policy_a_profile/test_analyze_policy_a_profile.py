#!/usr/bin/env python3
"""CPU-only tests for the fail-closed policy_a profile analyzer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "analyze_policy_a_profile.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "policy_a_profile_analyzer_tests", ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A = load_analyzer()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_trace() -> dict[str, object]:
    events: list[dict[str, object]] = [
        {
            "ph": "X",
            "cat": "Trace",
            "name": "PyTorch Profiler (0)",
            "ts": 1000.0,
            "dur": 2000.0,
        },
        {
            "ph": "i",
            "cat": "Trace",
            "name": "Iteration Start: PyTorch Profiler",
            "ts": 1000.0,
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::mm",
            "ts": 1100.0,
            "dur": 10.0,
            "pid": 123,
            "tid": 7,
            "args": {
                "External id": 11,
                "Input Dims": [[4, 8], [8, 16]],
                "Input Strides": [[8, 1], [16, 1]],
                "Input type": ["float", "float"],
            },
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "Cijk_test",
            "ts": 1200.0,
            "dur": 50.0,
            "pid": 0,
            "args": {"device": 0, "stream": 1, "External id": 11, "correlation": 3},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "unknown_without_external_id",
            "ts": 2200.0,
            "dur": 50.0,
            "pid": 0,
            "args": {"device": 0, "stream": 1, "correlation": 4},
        },
    ]
    for step, left in ((7, 1000.0), (8, 2000.0)):
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"LUMEN_TRAIN_STEP#{step}",
                "ts": left,
                "dur": 1000.0,
                "pid": 123,
                "tid": 7,
            }
        )
        for index in range(8):
            start = left + 10.0 + index * 100.0
            for offset, name in enumerate(
                (
                    "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__",
                    "FSDP::root_pre_forward",
                    "FSDP::pre_backward",
                    "FSDP::root_post_backward_callback",
                )
            ):
                events.append(
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": name,
                        "ts": start + offset * 10.0,
                        "dur": 5.0,
                        "pid": 123,
                        "tid": 7,
                    }
                )
        for offset, name in enumerate(
            ("Optimizer.step#AdamW.step", "Optimizer.zero_grad#AdamW.zero_grad")
        ):
            events.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": name,
                    "ts": left + 900.0 + offset * 20.0,
                    "dur": 10.0,
                    "pid": 123,
                    "tid": 7,
                }
            )
    return {"traceEvents": events}


class TestCoreHelpers(unittest.TestCase):
    def test_interval_union(self):
        self.assertEqual(A.interval_union([]), 0.0)
        self.assertEqual(A.interval_union([(0, 2), (1, 3), (5, 7)]), 5.0)
        self.assertEqual(A.interval_union([(3, 3), (4, 2)]), 0.0)

    def test_command_normalization_and_contract(self):
        bf16 = "torchrun train.py --mode bf16 --num-layers-at-end-in-bf16 1 --seed 1"
        mxfp4 = (
            "torchrun train.py --mode mxfp4 --num-layers-at-end-in-bf16 1 "
            "--seed 1 --mxfp4-pack-qkv --mxfp4-fuse-swiglu"
        )
        self.assertEqual(A.normalize_command(bf16), A.normalize_command(mxfp4))
        passed, evidence = A.command_contract(bf16, mxfp4)
        self.assertTrue(passed, evidence)
        passed, _ = A.command_contract(bf16, mxfp4.replace("--seed 1", "--seed 2"))
        self.assertFalse(passed)

    def test_matrix_shape_and_categories(self):
        producer = A.Producer("aten::mm", [[32, 64], [64, 128]], None, None)
        self.assertEqual(A.matrix_shape(producer), "M=32,N=128,K=64")
        addmm = A.Producer("aten::addmm", [[128], [32, 64], [64, 128]], None, None)
        self.assertEqual(A.matrix_shape(addmm), "M=32,N=128,K=64")
        quantized_producer = A.Producer(
            "QuantizedLinearFunction", [[2, 8192, 4096]], None, None
        )
        quantized_producer.scope = "quantized_linear_forward"
        a4w4 = A.GPUEvent(
            0,
            1,
            "kernel",
            "gemm_a4w4_test",
            1,
            1,
            0,
            quantized_producer,
        )
        a4w4.shape = "M=16384,N=4096,K=4096"
        self.assertEqual(A.classify_event(a4w4, token_rows=16384), "a4w4_forward")
        unknown = A.GPUEvent(0, 1, "kernel", "unknown", None, None, 0)
        self.assertEqual(A.classify_event(unknown), "other")


class TestStreamingTrace(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "trace.json"
        write_json(self.path, synthetic_trace())

    def tearDown(self):
        self.directory.cleanup()

    def test_stdlib_and_optional_ijson_paths(self):
        expected = list(A._stdlib_trace_events(self.path))
        previous = A._ijson
        try:
            A._ijson = None
            self.assertEqual(list(A.iter_trace_events(self.path)), expected)

            class FakeIjson:
                @staticmethod
                def items(handle, prefix):
                    self.assertEqual(prefix, "traceEvents.item")
                    return iter(json.load(handle)["traceEvents"])

            A._ijson = FakeIjson()
            self.assertEqual(list(A.iter_trace_events(self.path)), expected)
        finally:
            A._ijson = previous

    def test_exact_trace_contract_and_missing_external_id(self):
        log = {
            "steps": [
                {"step": 7, "step_time_ms": 1.0},
                {"step": 8, "step_time_ms": 1.0},
            ]
        }
        trace = A.scan_trace(
            self.path,
            log,
            precision="bf16",
            token_rows=16384,
            cache={"selected_identities": {}},
        )
        self.assertTrue(trace["contract"]["all_pass"], trace["contract"])
        self.assertEqual(trace["annotations"]["root_forward_total"], 16)
        self.assertEqual(trace["annotations"]["root_post_backward_total"], 16)
        self.assertEqual(trace["missing_external_id"]["events_total"], 1)

    def test_trace_log_span_mismatch_is_rejected(self):
        log = {
            "steps": [
                {"step": 7, "step_time_ms": 4.0},
                {"step": 8, "step_time_ms": 4.0},
            ]
        }
        trace = A.scan_trace(
            self.path,
            log,
            precision="bf16",
            token_rows=16384,
            cache={"selected_identities": {}},
        )
        self.assertFalse(trace["contract"]["checks"]["trace_log_span_within_3ms"])


class TestTrainingLog(unittest.TestCase):
    def test_real_step_profiler_markers_are_required_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            profile_path = path.parent / "profile.txt"
            lines = [f"> Profiler armed: steps 7-8 -> {profile_path}"]
            lines.extend(
                f"step {step}/8 | loss 1.0 | grad_norm 1.0 | lr 0.1 | "
                "step_time_ms 10 | peak_mem_gib 2"
                for step in range(1, 9)
            )
            lines.extend(
                [
                    f"> Profiler wrote {profile_path}",
                    "| val_loss 1.0",
                    "Training complete after 8 steps.",
                ]
            )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = A.parse_training_log(path, 8)
            self.assertTrue(result["profile_marker_order_exact"], result)
            self.assertEqual(result["profile_marker_counts"], {"start": 1, "stop": 1})

            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    f"> Profiler wrote {profile_path}", "profiler_stop"
                ),
                encoding="utf-8",
            )
            result = A.parse_training_log(path, 8)
            self.assertFalse(result["profile_marker_order_exact"])
            self.assertEqual(result["profile_marker_counts"], {"start": 1, "stop": 0})

    def test_fallback_and_bad_teardown_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "step 1/1 | loss 1.0 | grad_norm 1.0 | lr 0.1 | "
                "step_time_ms 10 | peak_mem_gib 2\n"
                "backend failed, trying next fallback\n"
                "| val_loss 1.0\n"
                "Training complete after 1 steps.\n"
                "unexpected teardown text\n",
                encoding="utf-8",
            )
            result = A.parse_training_log(path, 1)
            self.assertTrue(result["failure_markers_before_completion"]["fallback"])
            self.assertFalse(result["post_completion"]["valid"])

    def test_allowlisted_teardown_and_route_flush(self):
        block = [
            "Traceback (most recent call last):",
            '  File "weakref.py", line 666, in _exitfunc',
            "  File x, in __call__",
            '  File "torch/library.py", in _del_library',
            "    _clear_torch_ops_cache()",
            '    namespace, name = qualname.split("::")',
            "ValueError: too many values to unpack (expected 2)",
        ]
        lines = ["Training complete after 1 steps."]
        for rank in range(16):
            lines.extend(f"[rank{rank % 8}]: {line}" for line in block)
        lines.extend(
            [
                "lumen.ops.quantize.mxfp4_autotune:MXFP4 shape log: wrote 9 distinct shapes to /tmp/mxfp4-shapes-rank0.csv",
                "INFO:lumen.ops.quantize.mxfp4_autotune:MXFP4 shape log: wrote 9 distinct shapes to /tmp/mxfp4-shapes-rank0.csv",
            ]
        )
        result = A.analyze_post_completion(lines, 0, True)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["traceback_count"], 16)


class TestIntegrityHelpers(unittest.TestCase):
    def test_analyzer_test_is_frozen_in_source_audit_and_meta(self):
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness_names = (
                "run_policy_a_profile.sh",
                "route_entry.py",
                "analyze_policy_a_profile.py",
                "test_analyze_policy_a_profile.py",
                "protocol.md",
                "test_runner.py",
            )
            for name in harness_names:
                (root / name).write_text(name, encoding="utf-8")
            source_audit = root / "source-audit.json"
            audit = {
                "schema": 1,
                "lumen_branch": "dev/mxfp4",
                "lumen_commit": "b" * 40,
                "aiter_commit": "c" * 40,
                "lumen_tree_sha256": digest,
                "aiter_tree_sha256": digest,
                "source_bundle_sha256": digest,
                "runtime_modules_sha256": digest,
                "f4gemm_directory_sha256": digest,
                "tuned_tables_sha256": digest,
                "workload_sha256": digest,
                "imports": {
                    "lumen": "/home/xdai/Lumen/lumen/__init__.py",
                    "aiter": "/home/xdai/aiter/aiter/__init__.py",
                },
                "runtime": {"devices": 8, "torch": "test", "hip": "test"},
                "harness_sha256": {
                    str(root / name): A.sha256_file(root / name)
                    for name in harness_names
                },
            }
            write_json(source_audit, audit)
            source_ok, _ = A._validate_source_audit(root, audit)
            self.assertTrue(source_ok)

            meta = dict(A.EXPECTED_META)
            meta.update(
                {
                    "cache_sha256": digest,
                    "source_audit_sha256": A.sha256_file(source_audit),
                    "smoke_aiter_config_cache_sha256": "empty",
                    "runner_sha256": A.sha256_file(root / "run_policy_a_profile.sh"),
                    "route_entry_sha256": A.sha256_file(root / "route_entry.py"),
                    "analyzer_sha256": A.sha256_file(root / "analyze_policy_a_profile.py"),
                    "analyzer_test_sha256": A.sha256_file(
                        root / "test_analyze_policy_a_profile.py"
                    ),
                    "protocol_sha256": A.sha256_file(root / "protocol.md"),
                    "test_sha256": A.sha256_file(root / "test_runner.py"),
                }
            )
            meta_ok, _ = A._validate_profile_meta(root, meta, audit)
            self.assertTrue(meta_ok)

            meta["analyzer_test_sha256"] = "0" * 64
            meta_ok, _ = A._validate_profile_meta(root, meta, audit)
            self.assertFalse(meta_ok)

    def test_progress_contract_is_mode_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.log"
            base = [
                f"2026-01-01T00:00:0{index}Z stage={stage}"
                for index, stage in enumerate(A.EXPECTED_PROGRESS)
            ]
            path.write_text("\n".join(base) + "\n", encoding="utf-8")
            self.assertTrue(A._validate_progress(path, require_completion=False)[0])
            self.assertFalse(A._validate_progress(path, require_completion=True)[0])
            path.write_text(
                "\n".join([*base, "2026-01-01T00:00:09Z stage=manifest_ready"]) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(A._validate_progress(path, require_completion=False)[0])
            self.assertTrue(A._validate_progress(path, require_completion=True)[0])

    def test_status_kfd_and_source_records(self):
        digest = "a" * 64
        keys = (
            "lumen_commit",
            "aiter_commit",
            "lumen_tree_sha256",
            "aiter_tree_sha256",
            "source_bundle_sha256",
            "runtime_modules_sha256",
            "f4gemm_directory_sha256",
            "tuned_tables_sha256",
        )
        audit = {key: digest for key in keys}
        audit["workload_sha256"] = digest
        run_meta = {key: digest for key in keys}
        run_meta["workload_sha256"] = digest
        tree = {key: digest for key in keys}
        passed, _ = A.validate_source_records(audit, run_meta, tree, digest, digest)
        self.assertTrue(passed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status.txt"
            status.write_text("one=0\ntwo=0\n", encoding="utf-8")
            self.assertTrue(A.status_is_zero(status, ("one", "two")))
            kfd = root / "kfd.txt"
            kfd.write_text("2026-01-01\nstatus=idle\n", encoding="utf-8")
            self.assertTrue(A.validate_kfd_file(kfd)[0])
            kfd.write_text(
                "non_service_kfd_client pid=3\nstatus=busy\n", encoding="utf-8"
            )
            self.assertFalse(A.validate_kfd_file(kfd)[0])

    def test_cache_hash_and_identity_validation(self):
        payload = {
            "schema": 6,
            "arch": "gfx950",
            "choices": {"1,2,3": "asm"},
            "profiles": {
                "1,2,3": {
                    "identities": {
                        "asm": {
                            "implementation": "asm",
                            "kernel_name": "kernel",
                            "tile_m": 32,
                            "tile_n": 64,
                            "log2_k_split": 0,
                            "split_k_capable": False,
                            "manifest_sha256": "a" * 64,
                            "code_object": "kernel.co",
                            "code_object_sha256": "b" * 64,
                        }
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_json(path, payload)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(A.load_cache(path, digest)["valid"])
            bad = A.load_cache(path, "0" * 64)
            self.assertFalse(bad["valid"])
            self.assertIn("sha256_mismatch", bad["errors"])

    def test_route_exact_shape_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "smoke_mxfp4_policy_a"
            arm.mkdir()
            choices = {shape: "shuffled" for shape in A.EXPECTED_ROUTE_SHAPES}
            cache = {"choices": choices}
            names = [f"model.layers.35.linear{index}" for index in range(7)]
            names.append("lm_head")
            for rank in range(8):
                report = {
                    "counts": {
                        "rank": rank,
                        "world_size": 8,
                        "instrumentation_installed": 1,
                        "enabled_qkv": 35,
                        "enabled_swiglu": 35,
                        "qkv_linear_success": 1400,
                        "swiglu_fwd_success": 1400,
                        "swiglu_bwd_success": 840,
                    },
                    "unquantized_linear_names": names,
                    "lm_head_count": 1,
                    "lm_head_weight_dtype": "bfloat16",
                    "lm_head_quant_enabled": False,
                    "lumen_import": "/home/xdai/Lumen/lumen/__init__.py",
                    "aiter_import": "/home/xdai/aiter/aiter/__init__.py",
                }
                write_json(arm / f"route-rank{rank}.json", report)
                rows = ["M,N,K,asm_available,backend,calls"]
                rows.extend(
                    f"{shape},1,shuffled,{calls}"
                    for shape, calls in A.EXPECTED_ROUTE_SHAPES.items()
                )
                (arm / f"mxfp4-shapes-rank{rank}.csv").write_text(
                    "\n".join(rows) + "\n", encoding="utf-8"
                )
            result = A._route_arm(root, "smoke_mxfp4_policy_a", cache)
            self.assertTrue(result["valid"], result["errors"])
            path = arm / "mxfp4-shapes-rank7.csv"
            path.write_text(path.read_text().replace(",840\n", ",839\n", 1))
            self.assertFalse(A._route_arm(root, "smoke_mxfp4_policy_a", cache)["valid"])


if __name__ == "__main__":
    unittest.main()
