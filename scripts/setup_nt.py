"""Wire loop_dedup and loop_muon_hook into nexttransformer's ReferenceLM. Idempotent.

The GitHub repo yanghu819/loop-muon stays the single source of truth: loop_dedup.py, loop_muon.py,
and loop_muon_hook.py are copied in verbatim, never re-implemented here. Baseline behaviour is
bit-for-bit identical when LOOP_DEDUP_GAMMA and LOOP_MUON_GAMMA are unset (or 0).
"""
import argparse
import py_compile
import shutil
import sys
from pathlib import Path

IMPORT_ANCHOR = (
    "from nexttransformer.models.residual import HyperConnection, ManifoldHyperConnection, depth_attention"
)

HELPER = '''
# --- loop-dedup & loop-muon (github.com/yanghu819/loop-muon) ------------------------------------
import os as _os
from nexttransformer.methods.loop_dedup import LoopDedup as _LoopDedup
from nexttransformer.methods import loop_muon_hook as _loop_muon_hook

_DEDUP_CACHE = {}


def _loop_dedup():
    """None unless LOOP_DEDUP_GAMMA > 0, so the baseline path is untouched."""
    gamma = float(_os.environ.get("LOOP_DEDUP_GAMMA", "0"))
    if gamma <= 0:
        return None
    key = (gamma, _os.environ.get("LOOP_DEDUP_WINDOW", "8"), _os.environ.get("LOOP_DEDUP_RIDGE", "1e-2"))
    if key not in _DEDUP_CACHE:
        _DEDUP_CACHE[key] = _LoopDedup(gamma=gamma, window=int(key[1]), ridge=float(key[2]))
    return _DEDUP_CACHE[key]
# ------------------------------------------------------------------------------------------------
'''

INIT_ANCHOR = """        for name in self._frozen_native():
            getattr(self,name).requires_grad_(False)"""

INIT_NEW = """        for name in self._frozen_native():
            getattr(self,name).requires_grad_(False)
        if _loop_muon_hook.enabled() and config.method in ("huginn","ut_act","ouro"):
            _inc_out=bool(int(_os.environ.get("LOOP_MUON_INCLUDE_OUT","0")))
            _n_wrapped=_loop_muon_hook.wrap_core(
                self,config.pre_layers,config.pre_layers+config.core_layers-1,include_out=_inc_out)
            if _n_wrapped<=0:
                raise RuntimeError("loop_muon_hook wrapped 0 matrices in core")
            self._loop_muon_wrapped=_n_wrapped"""

LOOP_ANCHOR = """        gates=[]; logits_list=[]; losses=[]; masses=[]
        for r in range(n):"""

LOOP_NEW = """        gates=[]; logits_list=[]; losses=[]; masses=[]
        _dedup=_loop_dedup(); _dedup_hist=None; _dedup_frac=[]
        # Clear the history where the backward window starts; -1 means "never" (all loops carry grad).
        _dedup_reset=(n-c.huginn_backprop_loops) if c.method=="huginn" else -1
        _lm_on=_loop_muon_hook.enabled() and not is_gate_stage and torch.is_grad_enabled()
        _lm_start=(n-c.huginn_backprop_loops) if c.method=="huginn" else 0
        if _lm_on:
            _loop_muon_hook.begin(max(n-_lm_start,0))
        for r in range(n):"""

BLOCK_ANCHOR = """                for j in range(c.core_layers):
                    x=self._block(x,x0,idx,c.pre_layers+j,c.pre_layers+r*c.core_layers+j,total,
                                  alive=alive if c.method=="ut_act" else None)"""

BLOCK_NEW = """                _dedup_in=x
                if _lm_on:
                    _loop_muon_hook.set_loop((r-_lm_start) if not no_grad else None)
                try:
                    for j in range(c.core_layers):
                        x=self._block(x,x0,idx,c.pre_layers+j,c.pre_layers+r*c.core_layers+j,total,
                                      alive=alive if c.method=="ut_act" else None)
                finally:
                    if _lm_on:
                        _loop_muon_hook.set_loop(None)
                if _dedup is not None:
                    if r==_dedup_reset: _dedup_hist=None
                    _u_raw=x-_dedup_in
                    _u,_dedup_hist=_dedup(_u_raw,_dedup_hist,
                                          active=alive.to(_u_raw.dtype) if c.method=="ut_act" else None)
                    with torch.no_grad():
                        _num=(_u_raw-_u).float().pow(2).sum(-1)
                        _den=_u_raw.float().pow(2).sum(-1).clamp_min(1e-12)
                        _dedup_frac.append((_num/_den).mean())
                    x=_dedup_in+_u"""

STATS_ANCHOR = """        for j in range(c.post_layers):
            x=self._block(x,x0,idx,c.pre_layers+c.core_layers+j,c.pre_layers+n*c.core_layers+j,total)
        return self._readout(x,prelude),auxiliary,None"""

STATS_NEW = """        if _dedup is not None and _dedup_frac:
            # (removed energy)/(raw energy) per loop = gamma^2 * (fraction of the update inside the
            # previous loops' span), so span fraction = value / gamma^2.
            self.last_stats.update(dedup_removed_fraction=[float(v) for v in _dedup_frac],
                                   dedup_gamma=_dedup.gamma)
        for j in range(c.post_layers):
            x=self._block(x,x0,idx,c.pre_layers+c.core_layers+j,c.pre_layers+n*c.core_layers+j,total)
        return self._readout(x,prelude),auxiliary,None"""


def patch_text(orig: str) -> str:
    text = orig
    for anchor, new, label in (
        (IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + HELPER, "helper"),
        (INIT_ANCHOR, INIT_NEW, "init wrap_core"),
        (LOOP_ANCHOR, LOOP_NEW, "loop setup"),
        (BLOCK_ANCHOR, BLOCK_NEW, "core loop hooks"),
        (STATS_ANCHOR, STATS_NEW, "stats"),
    ):
        count = text.count(anchor)
        if count != 1:
            sys.exit(f"anchor for {label} appears {count} times, expected 1")
        text = text.replace(anchor, new)
        print(f"patched: {label}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--nt", type=Path, default=Path(os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer")))
    parser.add_argument("--dry-run", action="store_true", help="verify anchors without writing files")
    args = parser.parse_args()

    methods = args.nt / "nexttransformer/methods"
    lm = args.nt / "nexttransformer/models/lm.py"
    bak = args.nt / "nexttransformer/models/lm.py.orig"

    files = ("loop_dedup.py", "loop_muon.py", "loop_muon_hook.py")
    for name in files:
        src = args.src / name
        if not src.exists():
            sys.exit(f"missing {src}: sync github.com/yanghu819/loop-muon into {args.src} first")

    orig_path = bak if bak.exists() else lm
    patched = patch_text(orig_path.read_text())
    if args.dry_run:
        compile(patched, str(lm), "exec")
        print("dry-run ok: all 5 anchors matched and patched lm.py parses cleanly")
        return

    for name in files:
        shutil.copyfile(args.src / name, methods / name)
        print(f"copied {args.src / name} -> {methods / name}")

    if not bak.exists():
        shutil.copyfile(lm, bak)
        print(f"backed up {lm} -> {bak}")

    lm.write_text(patched)
    py_compile.compile(str(lm), doraise=True)
    print(f"{lm} patched and compiles")


if __name__ == "__main__":
    main()
