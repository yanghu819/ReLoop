"""The hook must produce exactly the gradient loop_gradient defines, and leave the forward alone."""

import torch
from torch import nn

import loop_muon_hook as hook
from loop_muon import loop_gradient

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


def build(d_in, d_out, s, batch, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(d_in, d_out, bias=False, dtype=F64)
    xs = [torch.randn(*batch, d_in, dtype=F64) for _ in range(s)]
    weights = [torch.randn(*batch, d_out, dtype=F64) for _ in range(s)]
    return lin, xs, weights


def test_hook_matches_loop_gradient():
    hook.CTX.gamma, hook.CTX.eps = 0.5, 1e-2
    d_in, d_out, s = 12, 9, 4
    lin, xs, weights = build(d_in, d_out, s, (2, 5))
    lin._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    lin.forward = type(lin).forward.__get__(lin)          # start from the plain forward
    import types
    lin.forward = types.MethodType(hook._hooked_forward, lin)

    ys = tied_run(lin, xs, weights)
    got = lin.weight.grad.clone()

    D = torch.stack([y.grad for y in ys], -1)             # [*batch, out, s]
    H = torch.stack(xs, -1)                               # [*batch, in,  s]
    want = loop_gradient(D, H, gamma=0.5, eps=1e-2).to(F64)

    # Both sides contract in fp32 on purpose, so they agree to fp32, not to fp64.
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 1e-6, f"hook gradient differs from the definition by {err:.2e}"
    print(f"hook reproduces loop_gradient to fp32 (relative error {err:.1e})")


def test_forward_is_unchanged():
    """The method acts only on the gradient; activations must match the baseline bit for bit."""
    import types
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
    import types
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


def test_buffers_are_released():
    """Nothing may stay pinned after the backward, or every microbatch leaks its activations."""
    import types
    d_in, d_out, s = 8, 8, 4
    lin, xs, weights = build(d_in, d_out, s, (2, 3), seed=3)
    lin._loop_muon_group = "g"
    hook.CTX.members = {"g": 1}
    hook.CTX.gamma = 0.5
    lin.forward = types.MethodType(hook._hooked_forward, lin)
    tied_run(lin, xs, weights)
    assert not hook.CTX.xs and not hook.CTX.op and not hook.CTX.pending, \
        f"left behind xs={len(hook.CTX.xs)} op={len(hook.CTX.op)} pending={len(hook.CTX.pending)}"
    print("all per-loop buffers are released once the backward finishes")


def test_shared_group_solves_once():
    """q/k/v read the same input, so the operator must be built once for the three of them."""
    import types
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
    real = hook.loop_operator

    def counting(*a, **k):
        built.append(1)
        return real(*a, **k)

    hook.loop_operator = counting
    try:
        hook.begin(s)
        ys = []
        for t, x in enumerate(xs):
            hook.set_loop(t)
            ys += [m(x) for m in mods]
        hook.set_loop(None)
        sum(y.sum() for y in ys).backward()
    finally:
        hook.loop_operator = real
    assert len(built) == 1, f"built the operator {len(built)} times for one shared input"
    assert not hook.CTX.xs, "shared group was not released"
    print("one operator serves all three matrices that read the same input")


if __name__ == "__main__":
    test_hook_matches_loop_gradient()
    test_forward_is_unchanged()
    test_disabled_is_the_ordinary_gradient()
    test_buffers_are_released()
    test_shared_group_solves_once()
    print("\nall checks passed")
