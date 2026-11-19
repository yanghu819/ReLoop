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


def test_a_finished_sequence_does_not_disturb_a_running_one():
    """Masking is per sequence: one batch row halting must not change another row's output."""
    torch.manual_seed(3)
    d, steps = 12, 6
    ups = [torch.randn(2, d, dtype=torch.float64) for _ in range(steps)]

    dedup, hist, both = LoopDedup(gamma=0.5, window=3), None, []
    for t, u in enumerate(ups):                      # row 1 halts after the second loop
        active = torch.tensor([1.0, 1.0 if t < 2 else 0.0], dtype=torch.float64)
        o, hist = dedup(u, hist, active=active)
        both.append(o[0])

    dedup, hist, alone = LoopDedup(gamma=0.5, window=3), None, []
    for u in ups:                                    # the same row, on its own, never masked
        o, hist = dedup(u[:1], hist)
        alone.append(o[0])

    err = max((a - b).abs().max().item() for a, b in zip(both, alone))
    assert err < 1e-12, f"a halted neighbour changed the live row by {err}"
    print("a halted sequence leaves its neighbour's output untouched")


def test_history_length_follows_the_backward_window():
    """Pins the reset itself, and the window it implies (see LoopDedup.window)."""
    seen = []

    class Recording(LoopDedup):
        def forward(self, update, history=None, active=None):
            seen.append(0 if history is None else len(history.updates))
            return super().forward(update, history, active)

    torch.manual_seed(0)
    block = torch.nn.Linear(16, 16, dtype=torch.float64)
    loop_forward(block, torch.randn(2, 16, dtype=torch.float64), num_loops=8,
                 backprop_depth=4, dedup=Recording(gamma=0.5))
    assert seen == [0, 1, 2, 3, 0, 1, 2, 3], seen
    print(f"history lengths {seen}: reset at the boundary, never longer than backprop_depth-1")


def test_evaluation_builds_no_graph():
    """enable_grad() would override a caller's no_grad and silently build a graph at eval."""
    with torch.no_grad():
        out = loop_forward(torch.nn.Linear(16, 16, dtype=torch.float64),
                           torch.randn(2, 16, dtype=torch.float64),
                           num_loops=8, backprop_depth=4, dedup=LoopDedup())
    assert not out.requires_grad, "evaluation under no_grad still built an autograd graph"
    print("evaluation under no_grad builds no graph")


def test_detaching_the_history_changes_the_gradient():
    """The method's central claim is backward == forward; this pins that it is not free."""
    def run(detach):
        torch.manual_seed(1)
        block, dedup = torch.nn.Linear(16, 16, dtype=torch.float64), LoopDedup(gamma=0.5)
        h, hist = torch.randn(2, 16, dtype=torch.float64), None
        for _ in range(4):
            u = block(h)
            if detach and hist is not None:
                hist = type(hist)(updates=[x.detach() for x in hist.updates],
                                  gram=hist.gram.detach())
            u, hist = dedup(u, hist)
            h = h + u
        h.sum().backward()
        return block.weight.grad.clone()

    attached, detached = run(False), run(True)
    rel = (attached - detached).abs().max().item() / attached.abs().max().item()
    assert rel > 1e-3, f"detaching the history barely moved the gradient ({rel})"
    print(f"detaching the history moves the weight gradient by {rel:.0%}")


def test_ridge_ignores_empty_slots():
    """A dead slot must not shrink the ridge for the live history."""
    torch.manual_seed(4)
    d = 10
    live, dead = LoopDedup(gamma=0.5, window=4), LoopDedup(gamma=0.5, window=4)
    u0, u1 = torch.randn(1, d, dtype=torch.float64), torch.randn(1, d, dtype=torch.float64)

    _, h = live(u0, None)
    want, _ = live(u1, h)

    _, h = dead(u0, None)
    _, h = dead(torch.randn(1, d, dtype=torch.float64), h,
                active=torch.zeros(1, dtype=torch.float64))   # one dead slot joins the window
    got, _ = dead(u1, h)

    err = (want - got).abs().max().item()
    assert err < 1e-12, f"an empty slot changed the live projection by {err}"
    print("an empty window slot leaves the ridge and the projection alone")


if __name__ == "__main__":
    test_matches_definition()
    test_removes_repetition_and_keeps_novelty()
    test_gradient_flows_through_the_projection()
    test_inactive_positions_do_not_evict_history()
    test_loop_forward_resets_at_the_backward_boundary()
    test_a_finished_sequence_does_not_disturb_a_running_one()
    test_history_length_follows_the_backward_window()
    test_evaluation_builds_no_graph()
    test_detaching_the_history_changes_the_gradient()
    test_ridge_ignores_empty_slots()
    print("\nall checks passed")
