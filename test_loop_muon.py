"""Checks the three properties loop_muon.py claims: inert when untied, descent-aligned, and acting
on the loop-varying part rather than on the shared content direction."""

import torch

from loop_muon import loop_apply, loop_gradient, loop_operator, loop_whiten

F64 = torch.float64


def test_untied_layer_is_bit_for_bit_identity():
    """s=1 is an ordinary layer used once. The method must not touch it at all."""
    H = torch.randn(7, 32, 1, dtype=F64)
    out = loop_whiten(H, gamma=0.5)
    assert torch.equal(out, H), (out - H).abs().max()
    # And through the gradient, for any backprop signal.
    D = torch.randn(7, 16, 1, dtype=F64)
    # loop_gradient contracts in fp32 on purpose, so it matches the plain gradient to fp32, not
    # bit for bit. The bit-for-bit claim is about loop_whiten, asserted above.
    got = loop_gradient(D, H)
    want = (D @ H.transpose(-1, -2)).sum(0).float()
    assert (got - want).abs().max() < 1e-5 * want.abs().max(), (got - want).abs().max()
    print("untied (s=1) layer: identity, bit for bit")


def test_descent_alignment_holds():
    """<grad~, grad> >= 0 per token, for any backprop signal: the step never opposes the gradient."""
    torch.manual_seed(0)
    worst = float("inf")
    for _ in range(300):
        s, d, o = int(torch.randint(2, 6, ())), 24, 18
        drift = torch.randn(d, 1, dtype=F64)
        # A deliberately nasty case: strong shared content plus nearly repeated updates.
        H = torch.randn(d, 1, dtype=F64) * 8 + drift * torch.arange(1., s + 1, dtype=F64)
        H = H + 0.01 * torch.randn(d, s, dtype=F64)
        D = torch.randn(o, s, dtype=F64) * torch.randn(1, s, dtype=F64).abs()
        G = D @ H.T
        Gt = D @ loop_whiten(H.unsqueeze(0), gamma=0.5).squeeze(0).T
        worst = min(worst, (Gt * G).sum().item() / G.norm().item() ** 2)
    assert worst >= -1e-12, f"descent alignment violated, worst normalised overlap {worst}"
    print(f"descent alignment holds over 300 trials (worst normalised overlap {worst:+.4f})")


def test_gram_of_whitened_against_raw_is_symmetric_psd():
    """The algebra behind the guarantee: H^T H~ = A_par + A_perp M is symmetric PSD."""
    torch.manual_seed(1)
    H = torch.randn(40, 5, dtype=F64) + torch.randn(40, 1, dtype=F64) * 5
    Ht = loop_whiten(H.unsqueeze(0), gamma=0.5).squeeze(0)
    K = H.T @ Ht
    assert (K - K.T).abs().max() < 1e-10, f"not symmetric: {(K - K.T).abs().max()}"
    assert torch.linalg.eigvalsh((K + K.T) / 2).min() > -1e-10, torch.linalg.eigvalsh(K)
    print("H^T H~ is symmetric PSD, which is what forces descent alignment")


def test_loop_index_split_would_break_descent():
    """Why the split is in feature space: centring along the loop index instead loses the guarantee.

    The counterexample is the one that killed the first version of this rule.
    """
    H = torch.tensor([[1.0, 2.0]], dtype=F64)          # d=1, s=2
    D = torch.tensor([[3.0, -1.0]], dtype=F64)
    s = 2
    P = torch.eye(s, dtype=F64) - torch.full((s, s), 1.0 / s, dtype=F64)
    Q = torch.full((s, s), 1.0 / s, dtype=F64)
    bad = H @ (Q + P * 3.0)                            # loop-index split, mu = 3
    assert torch.allclose(bad, torch.tensor([[0.0, 3.0]], dtype=F64)), bad
    assert ((D @ bad.T) * (D @ H.T)).sum().item() < 0, "expected the counterexample to be negative"
    good = loop_whiten(H.unsqueeze(0), gamma=0.5).squeeze(0)
    assert ((D @ good.T) * (D @ H.T)).sum().item() >= 0
    print("loop-index split gives grad=+1 but grad~=-3; the feature-space split stays aligned")


def test_shared_content_direction_is_untouched():
    """The loop-mean direction carries the token's content and must survive unchanged."""
    torch.manual_seed(2)
    H = torch.randn(32, 4, dtype=F64) + torch.randn(32, 1, dtype=F64) * 10
    Ht = loop_whiten(H.unsqueeze(0), gamma=1.0).squeeze(0)
    e = H.mean(-1)
    e = e / e.norm()
    before, after = e @ H, e @ Ht
    assert (before - after).abs().max() < 1e-10, (before, after)
    print("the loop-mean direction passes through untouched, even at gamma=1")


def test_identical_loops_are_a_no_op_and_repeated_updates_are_not():
    """No loop-to-loop motion means nothing to whiten; a repeated update is exactly the target."""
    h = torch.randn(16, 1, dtype=F64)
    frozen = h.repeat(1, 4)
    assert (loop_whiten(frozen.unsqueeze(0), gamma=0.5).squeeze(0) - frozen).abs().max() < 1e-10

    u = torch.randn(16, 1, dtype=F64) * 0.1               # the same update every loop
    repeated = h + u * torch.arange(4, dtype=F64)
    out = loop_whiten(repeated.unsqueeze(0), gamma=0.5).squeeze(0)
    moved = (out - repeated).norm() / repeated.norm()
    assert moved > 1e-3, f"a repeated update should be reweighted, moved only {moved}"
    print(f"identical loops: no-op; repeated update: reweighted ({moved:.1%} of the norm)")


def test_orthogonal_updates_are_nearly_untouched():
    """With no redundancy to remove the operator should be close to the identity.

    Measured against the loop-varying part, not against H. H_perp is only ~2% of H here, so a
    tolerance taken on ||H|| would also be met by M = 0, i.e. by deleting the very thing under test.
    """
    torch.manual_seed(3)
    d, s = 64, 4
    basis = torch.linalg.qr(torch.randn(d, s, dtype=F64))[0]
    H = torch.randn(d, 1, dtype=F64) * 3 + basis * 0.5     # mutually orthogonal, equal norm
    out = loop_whiten(H.unsqueeze(0), gamma=0.5).squeeze(0)

    e = H.mean(-1)
    e = e / e.norm()
    perp = H - torch.outer(e, e @ H)
    moved = (out - H).norm() / perp.norm()
    assert moved < 0.05, f"orthogonal loop inputs moved by {moved:.2%} of H_perp"
    # The threshold has to be able to fail: deleting H_perp entirely must not pass it.
    deleted = (torch.outer(e, e @ H) - H).norm() / perp.norm()
    assert deleted > 0.05, "tolerance is vacuous: M = 0 would also pass"
    print(f"mutually orthogonal updates: near-identity "
          f"(moved {moved:.2%} of H_perp; deleting it would move {deleted:.0%})")


def test_scale_invariance():
    """The operator normalises by tau, so scaling the inputs must only scale the output."""
    torch.manual_seed(7)
    H = torch.randn(1, 24, 4, dtype=F64)
    base = loop_whiten(H, gamma=0.5)
    for k in (1e-5, 1e-2, 1e3):
        scaled = loop_whiten(H * k, gamma=0.5) / k
        err = (scaled - base).abs().max().item() / base.abs().max().item()
        assert err < 1e-9, f"scaling by {k:g} changed the operator by {err:.2e}"
    tiny = loop_whiten(H * 1e-5, gamma=0.5)
    ratio = tiny.norm().item() / (H * 1e-5).norm().item()
    print(f"scale invariant over 1e-5..1e3 (tiny inputs give ||H~||/||H|| = {ratio:.3f})")


def test_near_zero_loop_mean_keeps_alignment():
    """A token whose loop mean is ~0 must still get a real projector, so alignment survives."""
    torch.manual_seed(11)
    worst = float("inf")
    for _ in range(500):
        s = int(torch.randint(2, 5, ()))
        X = torch.randn(6, s, dtype=F64)
        X = X - X.mean(-1, keepdim=True)                  # loop mean exactly zero
        X = X + 1e-7 * torch.randn(6, 1, dtype=F64)       # then a sliver of mean back
        D = torch.randn(5, s, dtype=F64)
        G = D @ X.T
        Gt = D @ loop_whiten(X.unsqueeze(0), gamma=1.0).squeeze(0).T
        worst = min(worst, (Gt * G).sum().item() / G.norm().item() ** 2)
    assert worst >= -1e-12, f"alignment broke at a near-zero loop mean: {worst}"
    print(f"near-zero loop mean: alignment holds (worst {worst:+.4f})")


def test_alignment_is_per_token_not_summed():
    """Pins the limit of the guarantee, so the README claim cannot quietly widen again."""
    g1 = torch.tensor([10.0, 1.0], dtype=F64)
    t1 = torch.tensor([5.0, 1.0], dtype=F64)
    g2 = torch.tensor([-9.0, 0.0], dtype=F64)
    assert (t1 * g1).sum() > 0 and (g2 * g2).sum() > 0
    assert ((t1 + g2) * (g1 + g2)).sum() < 0
    print("alignment is per token: two aligned tokens can still sum to a misaligned batch")


def test_dead_loops_do_not_enter_the_gradient():
    torch.manual_seed(13)
    H = torch.randn(1, 12, 3, dtype=F64)
    D = torch.randn(1, 5, 3, dtype=F64)
    active = torch.tensor([[1.0, 1.0, 0.0]], dtype=F64)
    masked = loop_gradient(D, H, gamma=0.5, active=active)
    live = loop_gradient(D[..., :2], H[..., :2], gamma=0.5)
    err = (masked - live).abs().max().item()
    assert err < 1e-9, f"a dead loop still contributed {err:.3e} to the gradient"
    print("a dead loop contributes nothing to the gradient")


def test_masked_loops_are_untouched():
    torch.manual_seed(4)
    H = torch.randn(2, 20, 4, dtype=F64)
    active = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]], dtype=F64)
    out = loop_whiten(H, gamma=0.5, active=active)
    assert torch.equal(out[1, :, 2:], H[1, :, 2:]), "masked loops were modified"
    solo = loop_whiten(H[1:, :, :2], gamma=0.5)
    assert (out[1, :, :2] - solo[0]).abs().max() < 1e-10, "mask changed the live loops"
    print("masked loops pass through, and do not disturb the live ones")


def test_streaming_form_matches_the_stacked_one():
    """loop_operator + loop_apply is what a real model can afford; it must be the same operator."""
    torch.manual_seed(17)
    for s in (2, 4, 8):
        xs = [torch.randn(3, 5, 32, dtype=F64) for _ in range(s)]
        H = torch.stack(xs, -1)
        want = loop_whiten(H, gamma=0.5)
        e, coef, M = loop_operator(xs, gamma=0.5)
        for t in range(s):
            got = loop_apply(xs, t, e, coef, M)
            err = (got - want[..., t]).abs().max().item()
            assert err < 1e-10, f"s={s} loop {t}: streaming form differs by {err:.2e}"
    print("streaming form reproduces the stacked operator for s = 2, 4, 8")


def test_streaming_form_needs_no_stack():
    """The whole point: the operator's state is s*s + s + d per token, not d*s."""
    xs = [torch.randn(2, 64, dtype=F64) for _ in range(4)]
    e, coef, M = loop_operator(xs, gamma=0.5)
    assert e.shape == (2, 64) and coef.shape == (2, 4) and M.shape == (2, 4, 4)
    stacked = 2 * 64 * 4
    streamed = e.numel() + coef.numel() + M.numel()
    print(f"streaming state {streamed} vs a stacked [batch, d, s] of {stacked} elements")


def test_gradient_shape_and_sum():
    D = torch.randn(5, 9, 3, dtype=F64)
    H = torch.randn(5, 7, 3, dtype=F64)
    g = loop_gradient(D, H, gamma=0.5)
    assert g.shape == (9, 7), g.shape
    manual = sum(D[n] @ loop_whiten(H, gamma=0.5)[n].T for n in range(5)).float()
    assert (g - manual).abs().max() < 1e-5 * manual.abs().max()
    print("loop_gradient sums per-token outer products into one [out, in] matrix")


if __name__ == "__main__":
    test_untied_layer_is_bit_for_bit_identity()
    test_descent_alignment_holds()
    test_gram_of_whitened_against_raw_is_symmetric_psd()
    test_loop_index_split_would_break_descent()
    test_shared_content_direction_is_untouched()
    test_identical_loops_are_a_no_op_and_repeated_updates_are_not()
    test_orthogonal_updates_are_nearly_untouched()
    test_scale_invariance()
    test_near_zero_loop_mean_keeps_alignment()
    test_alignment_is_per_token_not_summed()
    test_dead_loops_do_not_enter_the_gradient()
    test_streaming_form_matches_the_stacked_one()
    test_streaming_form_needs_no_stack()
    test_masked_loops_are_untouched()
    test_gradient_shape_and_sum()
    print("\nall checks passed")
