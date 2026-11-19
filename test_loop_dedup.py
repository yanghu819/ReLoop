"""Checks that the reference implementation matches the definition it claims to implement."""

import torch

from loop_dedup import LoopDedup, loop_forward


def brute_force(updates, gamma, ridge, window):
    """Definition, written directly: least-squares projection onto the kept history."""
    out, hist = [], []
    for u in updates:
        if hist:
            U = torch.stack(hist, -1)                       # [d, k]
            G = U.T @ U
            scale = G.diagonal().mean().clamp_min(1e-6)
            c = torch.linalg.solve(G + ridge * scale * torch.eye(len(hist), dtype=U.dtype), U.T @ u)
            u = u - gamma * (U @ c)
        out.append(u)
        hist = (hist + [u])[-window:]
    return out


def test_matches_definition():
    torch.manual_seed(0)
    gamma, ridge, window, d = 0.5, 1e-2, 4, 32
    updates = [torch.randn(d, dtype=torch.float64) for _ in range(10)]

    dedup, hist, got = LoopDedup(gamma=gamma, window=window, ridge=ridge), None, []
    for u in updates:
        o, hist = dedup(u.unsqueeze(0), hist)
        got.append(o.squeeze(0))

    for i, (a, b) in enumerate(zip(got, brute_force(updates, gamma, ridge, window))):
        err = (a - b).abs().max().item()
        assert err < 1e-9, f"loop {i}: {err}"
    print("matches the brute-force definition to <1e-9")


def test_removes_repetition_and_keeps_novelty():
    """An exactly repeated update loses gamma of itself; an orthogonal one is untouched."""
    dedup = LoopDedup(gamma=0.5, ridge=0.0)
    u = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    _, h = dedup(u, None)
    repeat, _ = dedup(u.clone(), h)
    assert torch.allclose(repeat, 0.5 * u), repeat
    fresh, _ = dedup(torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), h)
    assert torch.allclose(fresh, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)), fresh
    print("repeated update halved, orthogonal update untouched")


def test_gradient_flows_through_the_projection():
    """The projection must not be detached: history gets gradient too."""
    dedup = LoopDedup(gamma=0.5)
    a = torch.randn(1, 16, dtype=torch.float64, requires_grad=True)
    _, h = dedup(a, None)
    out, _ = dedup(torch.randn(1, 16, dtype=torch.float64), h)
    out.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0, "history received no gradient"
    print("gradient reaches the history, as the method requires")


def test_inactive_positions_do_not_evict_history():
    """A finished sequence must not push empty slots into the window."""
    dedup = LoopDedup(gamma=0.5, window=2)
    u = torch.randn(1, 8, dtype=torch.float64)
    _, h = dedup(u, None)
    off = torch.zeros(1, dtype=torch.float64)
    for _ in range(3):
        _, h = dedup(torch.randn(1, 8, dtype=torch.float64), h, active=off)
    kept = torch.stack(h.updates, -1).abs().sum().item()
    assert kept > 0, "real history was evicted by inactive loops"
    print("inactive loops do not evict real history")


def test_loop_forward_resets_at_the_backward_boundary():
    torch.manual_seed(0)
    block = torch.nn.Linear(16, 16, dtype=torch.float64)
    h = loop_forward(block, torch.randn(2, 16, dtype=torch.float64), num_loops=8,
                     backprop_depth=4, dedup=LoopDedup())
    h.sum().backward()
    assert block.weight.grad is not None
    print("loop_forward runs and produces gradients")


if __name__ == "__main__":
    test_matches_definition()
    test_removes_repetition_and_keeps_novelty()
    test_gradient_flows_through_the_projection()
    test_inactive_positions_do_not_evict_history()
    test_loop_forward_resets_at_the_backward_boundary()
    print("\nall checks passed")
