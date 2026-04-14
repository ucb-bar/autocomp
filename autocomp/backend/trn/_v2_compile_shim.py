"""Shim torch/XLA for NKI v2 compile-only scripts.

When imported and install_shim() is called, torch tensor creation and
xla_device() are replaced so that test_nki() functions produce numpy
arrays that CompileKernel(simulation=True) can consume.

CompileKernel returns numpy arrays; we provide NkiArray (a subclass of
np.ndarray) so .detach().cpu().numpy().to() chains work in test_nki.
"""

import sys
import types
import os
import shutil
import tempfile
import numpy as np


class NkiArray(np.ndarray):
    """np.ndarray subclass with torch-like .detach().cpu().numpy().to()."""
    def detach(self): return self
    def cpu(self): return self
    def to(self, *a, **kw): return self


class OnceCompileKernel:
    """Wraps a raw NKI kernel function so only the first call compiles.

    Subsequent calls return the cached result (compilation already
    captured the NEFF). This avoids neuronx-cc's 'artifacts already exist'
    error when test_nki calls the kernel multiple times.
    """
    def __init__(self, raw_fn, neff_dest):
        self._fn = raw_fn
        self._neff_dest = neff_dest
        self._result = None

    def __call__(self, *a, **kw):
        if self._result is not None:
            return self._result
        from nki.framework.compiled import CompileKernel
        ad = tempfile.mkdtemp(prefix="nki_compile_")
        ck = CompileKernel(func=self._fn, artifacts_dir=ad, _enable_simulation=True)
        self._result = ck(*a, **kw)
        cneff = os.path.join(ad, "kernel.neff")
        if os.path.exists(cneff):
            shutil.copy2(cneff, self._neff_dest)
        return self._result


def _as_nki(arr):
    """View a numpy array as NkiArray (zero-copy)."""
    return arr.view(NkiArray)


def install_shim():
    """Install torch/XLA fakes before the test preamble imports them."""

    # --- Fake torch_xla ---
    class _FakeXlaDevice:
        type = "xla"
        def __repr__(self):
            return "xla:0"

    class _FakeXm:
        @staticmethod
        def xla_device():
            return _FakeXlaDevice()

    fxla = types.ModuleType("torch_xla")
    fcore = types.ModuleType("torch_xla.core")
    fxm = types.ModuleType("torch_xla.core.xla_model")

    fxm.xla_device = _FakeXm.xla_device
    fcore.xla_model = fxm
    fxla.core = fcore
    fxla.device = _FakeXlaDevice

    sys.modules.setdefault("torch_xla", fxla)
    sys.modules.setdefault("torch_xla.core", fcore)
    sys.modules.setdefault("torch_xla.core.xla_model", fxm)

    # --- Make numpy arrays support .detach().cpu().numpy().to() chains ---
    # numpy.ndarray is a C type and can't be monkey-patched.
    # Instead, return our NkiArray subclass from all tensor creation shims.
    class NkiArray(np.ndarray):
        """np.ndarray subclass with torch-like .detach().cpu().numpy().to()."""
        def detach(self): return self
        def cpu(self): return self
        def to(self, *a, **kw): return self

    def _as_nki(arr):
        """View a numpy array as NkiArray (zero-copy)."""
        return arr.view(NkiArray)

    # --- Monkey-patch torch to return plain numpy arrays ---
    import torch

    _TORCH_TO_NP = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
    }

    def _np_dtype(torch_dtype):
        return _TORCH_TO_NP.get(torch_dtype, np.float32)

    def _fake_tensor(data, **kw):
        dt = kw.get("dtype")
        nd = _np_dtype(dt) if dt else None
        if isinstance(data, np.ndarray):
            return _as_nki(data.astype(nd) if nd else data.copy())
        return _as_nki(np.array(data, dtype=nd))

    def _fake_from_numpy(arr):
        return _as_nki(arr.copy())

    def _fake_randn(*shape, **kw):
        dt = kw.get("dtype")
        nd = _np_dtype(dt) if dt else np.float32
        return _as_nki(np.random.randn(*shape).astype(nd))

    torch.tensor = _fake_tensor
    torch.from_numpy = _fake_from_numpy
    torch.randn = _fake_randn
