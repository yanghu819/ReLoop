"""The hook must produce exactly the gradient loop_gradient defines, and leave the forward alone."""

import types

import torch
from torch import nn

import loop_muon_hook as hook
from loop_muon import loop_whiten

F64 = torch.float64


def tied_run(lin, xs, weights):
    """Run the shared matrix once per loop, keeping each loop's output for its backprop signal."""
    hook.begin(len(xs))
    ys = []
    for t, x in enumerate(xs):
        hook.set_loop(t)
        y = lin(x)
        y.retain_grad()
        ys.append(y)
    hook.set_loop(None)
    loss = sum((y * w).sum() for y, w in zip(ys, weights))
    loss.backward()
    return ys


def build(d_in, d_out, s, batch, seed=0, dtype=F64):
    torch.manual_seed(seed)
    lin = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
    xs = [torch.randn(*batch, d_in, dtype=dtype) for _ in range(s)]
    weights = [torch.randn(*batch, d_out, dtype=dtype) for _ in range(s)]
    return lin, xs, weights


def test_hook_matches_loop_whiten_exact_fp64():
    """In float64 the hooked weight gradient must match D @ loop_whiten(H)^T to <1e-11."""
    hook.CTX.gamma, hook.CTX.eps = 0.5, 1e-2
    for s in (2, 3, 4, 6):
        d_in, d_out = 12, 9
        lin, xs, weights = build(d_in, d_out, s, (2, 5), seed=s)
        lin._loop_muon_group = "g"
        hook.CTX.members = {"g": 1}
        lin.forward = types.MethodType(hook._hooked_forward, lin)

        ys = tied_run(lin, xs, weights)
        got = lin.weight.grad.clone()

        D = torch.stack([y.grad for y in ys], -1)         # [*batch, out, s]
        H = torch.stack(xs, -1)                           # [*batch, in,  s]
        Ht = loop_whiten(H, gamma=0.5, eps=1e-2)
        want = (D @ Ht.transpose(-1, -2)).flatten(0, -3).sum(0)

        err = (got - want).abs().max().item() / want.abs().max().item()
        assert err < 1e-11, f"s={s}: hook gradient differs from fp64 definition by {err:.2e}"
    print("hook reproduces exact fp64 loop_whiten gradient for s = 2, 3, 4, 6 (<1e-11)")


def test_bf16_delta_gemm_preserves_small_perp_signal():
    """With 2% perp/par ratio in bf16, split delta-GEMM keeps <2% error on the perp correction."""
    torch.manual_seed(9)
    B, T, d_in, d_out, s = 2, 64, 128, 96, 4
    base = torch.randn(B, T, d_in)
    u = torch.randn(B, T, d_in) * 0.02
    xs = [(base + (t - 1.5) * u + 0.002 * torch.randn(B, T, d_in)).bfloat16() for t in range(s)]
    ws = [torch.randn(B, T, d_out, dtype=torch.bfloat16) for _ in range(s)]

    # Master weight in fp32 (matching nexttransformer's Linear), activations in bf16.
    lin_base = nn.Linear(d_in, d_out, bias=False, dtype=torch.float32)
    lin_hook = nn.Linear(d_in, d_out, bias=False, dtype=torch.float32)
    lin_hook.weight.data.copy_(lin_base.weight.data)

    # Plain bf16 baseline gradient (same weight.to(bf16) forward)
    ys_b = [torch.nn.functional.linear(x, lin_base.weight.to(x.dtype)) for x in xs]
    for y in ys_b:
        y.retain_grad()
    sum((y * w).sum() for y, w in zip(ys_b, ws)).backward()
    g_base = lin_base.weight.grad.clone()

    # Hooked bf16 gradient
    lin_hook._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    hook.CTX.gamma, hook.CTX.eps = 0.5, 1e-2
    lin_hook.forward = types.MethodType(hook._hooked_forward, lin_hook)
    ys_h = tied_run(lin_hook, xs, ws)
    g_hook = lin_hook.weight.grad.clone()

    # Reference delta in fp64: sum_t d_t^T (H_tilde - H)_t
    D64 = torch.stack([y.grad.double() for y in ys_h], -1)
    H64 = torch.stack([x.double() for x in xs], -1)
    delta_H64 = loop_whiten(H64, gamma=0.5, eps=1e-2) - H64
    delta_want = (D64 @ delta_H64.transpose(-1, -2)).flatten(0, -3).sum(0).float()

    delta_got = g_hook - g_base
    rel = (delta_got - delta_want).norm().item() / delta_want.norm().item()
    assert rel < 0.02, f"bf16 split delta-GEMM lost too much precision on perp delta: {rel:.3%}"
    print(f"bf16 split delta-GEMM keeps perp correction accurate (rel error {rel:.2%})")


def test_forward_is_unchanged():
    """The method acts only on the gradient; activations must match the baseline bit for bit."""
    d_in, d_out, s = 10, 7, 3
    lin, xs, _ = build(d_in, d_out, s, (3, 4), seed=1)
    plain = [lin(x).detach().clone() for x in xs]

    lin._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    hook.CTX.gamma = 0.5
    lin.forward = types.MethodType(hook._hooked_forward, lin)
    hook.begin(s)
    for t, x in enumerate(xs):
        hook.set_loop(t)
        assert torch.equal(lin(x).detach(), plain[t]), f"forward changed at loop {t}"
    hook.set_loop(None)
    print("forward pass is bit-for-bit identical to the baseline")


def test_disabled_is_the_ordinary_gradient():
    d_in, d_out, s = 10, 6, 3
    lin, xs, weights = build(d_in, d_out, s, (2, 4), seed=2)
    tied_run(lin, xs, weights)
    plain_grad = lin.weight.grad.clone()

    lin2, xs2, w2 = build(d_in, d_out, s, (2, 4), seed=2)
    lin2._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    hook.CTX.gamma = 0.0                                  # disabled
    lin2.forward = types.MethodType(hook._hooked_forward, lin2)
    tied_run(lin2, xs2, w2)
    err = (lin2.weight.grad - plain_grad).abs().max().item()
    assert err == 0.0, f"disabled hook changed the gradient by {err:.2e}"
    hook.CTX.gamma = 0.5
    print("with gamma=0 the gradient is bit-for-bit the ordinary one")


def test_buffers_are_released_for_all_s():
    """Nothing may stay pinned after backward, including s=1 (single live loop)."""
    for s in (1, 4):
        d_in, d_out = 8, 8
        lin, xs, weights = build(d_in, d_out, s, (2, 3), seed=3 + s)
        lin._loop_muon_group = "g"
        hook.CTX.members = {"g": 1}
        hook.CTX.gamma = 0.5
        lin.forward = types.MethodType(hook._hooked_forward, lin)
        tied_run(lin, xs, weights)
        assert not hook.CTX.xs and not hook.CTX.op and not hook.CTX.pending, \
            f"s={s}: left behind xs={len(hook.CTX.xs)} op={len(hook.CTX.op)} pending={len(hook.CTX.pending)}"
    print("all per-loop buffers are released once the backward finishes (s = 1 and s = 4)")


def test_eval_no_grad_does_not_capture():
    """Evaluation under torch.no_grad() must neither capture activations nor build a graph."""
    d_in, d_out, s = 8, 8, 4
    lin, xs, _ = build(d_in, d_out, s, (2, 3), seed=5)
    lin._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    hook.CTX.gamma = 0.5
    lin.forward = types.MethodType(hook._hooked_forward, lin)
    hook.begin(s)
    with torch.no_grad():
        for t, x in enumerate(xs):
            hook.set_loop(t)
            y = lin(x)
            assert not y.requires_grad
    hook.set_loop(None)
    assert not hook.CTX.xs and not hook.CTX.op, "no_grad evaluation captured state"
    print("no_grad evaluation captures nothing and keeps requires_grad=False")


def test_shared_group_solves_once():
    """q/k/v read the same input, so the operator must be built once for the three of them."""
    d, s = 10, 4
    torch.manual_seed(4)
    mods = [nn.Linear(d, d, bias=False, dtype=F64) for _ in range(3)]
    for m in mods:
        m._loop_muon_group = "shared"
        m.forward = types.MethodType(hook._hooked_forward, m)
    hook.CTX.members = {"shared": 3}
    hook.CTX.gamma = 0.5

    xs = [torch.randn(2, 3, d, dtype=F64) for _ in range(s)]
    built = []
    real = hook.loop_delta_operator

    def counting(*a, **k):
        built.append(1)
        return real(*a, **k)

    hook.loop_delta_operator = counting
    try:
        hook.begin(s)
        ys = []
        for t, x in enumerate(xs):
            hook.set_loop(t)
            ys += [m(x) for m in mods]
        hook.set_loop(None)
        sum(y.sum() for y in ys).backward()
    finally:
        hook.loop_delta_operator = real
    assert len(built) == 1, f"built the operator {len(built)} times for one shared input"
    assert not hook.CTX.xs, "shared group was not released"
    print("one operator serves all three matrices that read the same input")


def test_grad_accumulation_across_microbatches():
    """Two microbatches accumulated before zero_grad must equal the sum of their individual gradients."""
    d_in, d_out, s = 10, 8, 4
    lin1, xs1, w1 = build(d_in, d_out, s, (2, 4), seed=21)
    _, xs2, w2 = build(d_in, d_out, s, (2, 4), seed=22)
    lin1._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    hook.CTX.gamma = 0.5
    lin1.forward = types.MethodType(hook._hooked_forward, lin1)

    tied_run(lin1, xs1, w1)
    g1 = lin1.weight.grad.clone()
    lin1.weight.grad = None
    tied_run(lin1, xs2, w2)
    g2 = lin1.weight.grad.clone()

    lin1.weight.grad = None
    tied_run(lin1, xs1, w1)
    tied_run(lin1, xs2, w2)
    g_acc = lin1.weight.grad.clone()

    assert torch.allclose(g_acc, g1 + g2, rtol=0, atol=1e-12)
    assert not hook.CTX.xs
    print("gradient accumulation across microbatches sums cleanly and leaks nothing")


if __name__ == "__main__":
    test_hook_matches_loop_whiten_exact_fp64()
    test_bf16_delta_gemm_preserves_small_perp_signal()
    test_forward_is_unchanged()
    test_disabled_is_the_ordinary_gradient()
    test_buffers_are_released_for_all_s()
    test_eval_no_grad_does_not_capture()
    test_shared_group_solves_once()
    test_grad_accumulation_across_microbatches()
    print("\nall checks passed")
