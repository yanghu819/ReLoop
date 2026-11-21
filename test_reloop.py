"""Unit tests for reloop.py (ReLoop vertical 2-scalar integrator & TapFeedbackGate)."""
import torch
from torch import nn

from reloop import ReLoop, TapFeedbackGate


def test_reloop_parameter_budget_is_two_floats():
    mod = ReLoop(alpha_init=4.0, beta_init=1.5)
    assert sum(p.numel() for p in mod.parameters()) == 2
    assert mod.alpha.shape == (1,) and mod.beta.shape == (1,)
    assert mod.alpha.dtype == torch.float32 and mod.beta.dtype == torch.float32


def test_reloop_exact_formula_and_gradients():
    torch.manual_seed(0)
    mod = ReLoop(alpha_init=3.2, beta_init=0.9)
    core = nn.Linear(16, 16, bias=False)
    x_pre = torch.randn(2, 5, 16, requires_grad=True)
    loops = 4

    out = mod(x_pre, lambda x, r: x + core(x), loops=loops)

    ref = x_pre
    for _ in range(loops):
        h = ref + core(ref)
        ref = ref + (1.0 / loops) * (mod.alpha * (h - ref) + mod.beta * x_pre)

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    out.square().mean().backward()
    assert mod.alpha.grad is not None and mod.alpha.grad.abs().item() > 0
    assert mod.beta.grad is not None and mod.beta.grad.abs().item() > 0
    assert x_pre.grad is not None and torch.isfinite(x_pre.grad).all()


def test_tbptt_at_or_above_loops_matches_full_bptt():
    torch.manual_seed(3)
    core = nn.Linear(12, 12, bias=False)
    base_x = torch.randn(2, 4, 12)
    loops = 4
    results = []
    for tbptt in (0, loops, loops + 2):
        mod = ReLoop(alpha_init=2.5, beta_init=0.8).train()
        core.zero_grad(set_to_none=True)
        x_pre = base_x.clone().detach().requires_grad_(True)
        out = mod(x_pre, lambda x, r: x + 0.1 * core(x), loops=loops, tbptt_loops=tbptt)
        out.square().mean().backward()
        results.append((out.detach(), x_pre.grad.clone(), mod.alpha.grad.clone(), mod.beta.grad.clone()))
    for r in results[1:]:
        for a, b in zip(results[0], r):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_reloop_causality_and_depth_extrapolation():
    torch.manual_seed(1)
    mod = ReLoop(alpha_init=4.0, beta_init=1.5).eval()
    core = nn.Linear(12, 12, bias=False)
    x = torch.randn(2, 6, 12)
    x_mut = x.clone()
    x_mut[:, -1] += 10.0

    for loops in (2, 4, 8, 16, 32):
        y1 = mod(x, lambda h, r: h + 0.05 * core(h), loops=loops)
        y2 = mod(x_mut, lambda h, r: h + 0.05 * core(h), loops=loops)
        torch.testing.assert_close(y1[:, :-1], y2[:, :-1], rtol=1e-6, atol=1e-6)
        assert torch.isfinite(y1).all()


def test_reloop_spherical_and_tbptt_and_bf16_dtype():
    torch.manual_seed(2)
    mod = ReLoop(alpha_init=4.0, beta_init=1.5, spherical=True).train()
    core = nn.Linear(16, 16, bias=False).to(torch.bfloat16)
    x_pre = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    out = mod(x_pre, lambda x, r: x + (0.1 * core(x)).to(x.dtype), loops=8, tbptt_loops=4)
    assert out.dtype == torch.bfloat16
    assert mod.alpha.dtype == torch.float32 and mod.beta.dtype == torch.float32
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=2e-2, atol=2e-2)
    out.float().sum().backward()
    assert mod.alpha.grad is not None and mod.beta.grad is not None
    assert core.weight.grad is not None and torch.isfinite(core.weight.grad).all()


def test_tap_feedback_gate_budget_scaling_and_causality():
    gate = TapFeedbackGate(dim=1536, alpha_init=2.0)
    assert sum(p.numel() for p in gate.parameters()) == 1
    assert gate.alpha.shape == (1,) and float(gate.alpha.item()) == 2.0
    x_emb = torch.randn(2, 6, 1536, dtype=torch.bfloat16) * 0.02
    fb = torch.randn(2, 6, 1536, dtype=torch.bfloat16) * 0.55
    out = gate(x_emb, fb)
    assert out.dtype == torch.bfloat16
    # Position 0 must remain untouched
    torch.testing.assert_close(out[:, 0], x_emb[:, 0], rtol=0, atol=0)
    # Strictly causal: mutating fb at token t cannot change out at positions <= t
    fb_mut = fb.clone()
    fb_mut[:, 3:] += 100.0
    out_mut = gate(x_emb, fb_mut)
    torch.testing.assert_close(out[:, :4], out_mut[:, :4], rtol=0, atol=0)
    # Verify amplitude calibration (O(0.028) vs embedding O(0.02), not 27.7x shock)
    pert_rms = (out[:, 1:] - x_emb[:, 1:]).float().pow(2).mean().sqrt().item()
    assert 0.01 < pert_rms < 0.06


if __name__ == "__main__":
    test_reloop_parameter_budget_is_two_floats()
    test_reloop_exact_formula_and_gradients()
    test_tbptt_at_or_above_loops_matches_full_bptt()
    test_reloop_causality_and_depth_extrapolation()
    test_reloop_spherical_and_tbptt_and_bf16_dtype()
    test_tap_feedback_gate_budget_scaling_and_causality()
    print("ALL RELOOP TESTS PASSED")
