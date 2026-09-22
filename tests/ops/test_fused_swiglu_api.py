import importlib.util
import sys
import types
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[2] / "lumen" / "ops" / "fused_swiglu.py"
_SPEC = importlib.util.spec_from_file_location("_lumen_fused_swiglu_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
swiglu_ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(swiglu_ops)


def _install_fake_activation(monkeypatch):
    calls = []
    activation = types.ModuleType("aiter.ops.triton.activation")

    def swiglu_fwd(x):
        calls.append(("forward", x))
        return x[..., : x.shape[-1] // 2]

    def swiglu_bwd(grad_output, x):
        calls.append(("backward", grad_output, x))
        return torch.cat((grad_output, grad_output), dim=-1)

    activation.swiglu_fwd = swiglu_fwd
    activation.swiglu_bwd = swiglu_bwd

    aiter = types.ModuleType("aiter")
    ops = types.ModuleType("aiter.ops")
    triton = types.ModuleType("aiter.ops.triton")
    aiter.ops = ops
    ops.triton = triton
    triton.activation = activation
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton", triton)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.activation", activation)
    return calls


def test_fused_swiglu_uses_public_aiter_forward(monkeypatch):
    calls = _install_fake_activation(monkeypatch)
    x = torch.arange(12).reshape(2, 6)

    out = swiglu_ops.fused_swiglu(x)

    assert calls == [("forward", x)]
    assert torch.equal(out, x[:, :3])


def test_fused_swiglu_backward_uses_public_aiter_backward(monkeypatch):
    calls = _install_fake_activation(monkeypatch)
    x = torch.arange(12).reshape(2, 6)
    grad_output = torch.ones(2, 6)[:, ::2]
    assert not grad_output.is_contiguous()

    out = swiglu_ops.fused_swiglu_backward(grad_output, x)

    assert calls == [("backward", grad_output, x)]
    assert torch.equal(out, torch.ones(2, 6))
