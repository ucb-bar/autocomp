"""libnrt ctypes wrapper for loading and executing pre-compiled NEFFs.

Provides NrtModel: a callable that loads a NEFF file via the Neuron Runtime
C library (libnrt) and executes it without any NKI recompilation. Used as a
drop-in replacement for @nki.jit functions in test_nki() correctness checks
and for latency benchmarking via nrt_execute loops.
"""

import ctypes
import numpy as np
import time

_libnrt = ctypes.CDLL("/opt/aws/neuron/lib/libnrt.so")

# ---- Struct definitions ----

class _NrtTensorInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("usage", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("dtype", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_uint32)),
        ("ndim", ctypes.c_uint32),
    ]

# ---- Function signatures ----

_libnrt.nrt_init.restype = ctypes.c_int
_libnrt.nrt_init.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
_libnrt.nrt_close.restype = ctypes.c_int
_libnrt.nrt_load.restype = ctypes.c_int
_libnrt.nrt_load.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
]
_libnrt.nrt_unload.restype = ctypes.c_int
_libnrt.nrt_unload.argtypes = [ctypes.c_void_p]
_libnrt.nrt_get_model_tensor_info.restype = ctypes.c_int
_libnrt.nrt_get_model_tensor_info.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
]
_libnrt.nrt_free_model_tensor_info.restype = ctypes.c_int
_libnrt.nrt_free_model_tensor_info.argtypes = [ctypes.c_void_p]
_libnrt.nrt_allocate_tensor_set.restype = ctypes.c_int
_libnrt.nrt_allocate_tensor_set.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libnrt.nrt_destroy_tensor_set.restype = None
_libnrt.nrt_destroy_tensor_set.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libnrt.nrt_tensor_allocate.restype = ctypes.c_int
_libnrt.nrt_tensor_allocate.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
]
_libnrt.nrt_tensor_free.restype = None
_libnrt.nrt_tensor_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libnrt.nrt_tensor_write.restype = ctypes.c_int
_libnrt.nrt_tensor_write.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
]
_libnrt.nrt_tensor_read.restype = ctypes.c_int
_libnrt.nrt_tensor_read.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
]
_libnrt.nrt_add_tensor_to_tensor_set.restype = ctypes.c_int
_libnrt.nrt_add_tensor_to_tensor_set.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
]
_libnrt.nrt_execute.restype = ctypes.c_int
_libnrt.nrt_execute.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

# ---- Constants ----

_NRT_SUCCESS = 0
_NRT_FRAMEWORK_TYPE_NO_FW = 1
_NRT_TENSOR_PLACEMENT_DEVICE = 0
_NRT_TENSOR_USAGE_INPUT = 0

# Map NRT dtype enum to numpy dtypes
def _build_dtype_map():
    m = {
        1: np.float32,
        2: np.float16,
        4: np.int8,
        5: np.uint8,
        6: np.int16,
        7: np.uint16,
        8: np.int32,
        9: np.uint32,
        10: np.int64,
        11: np.uint64,
    }
    try:
        import neuronxcc.nki.language as nl
        m[3] = nl.bfloat16
    except Exception:
        m[3] = np.uint16
    return m

_NRT_DTYPE_MAP = _build_dtype_map()

# ---- Runtime state ----

_nrt_initialized = False


def nrt_init():
    """Initialize the Neuron Runtime (idempotent)."""
    global _nrt_initialized
    if not _nrt_initialized:
        ret = _libnrt.nrt_init(_NRT_FRAMEWORK_TYPE_NO_FW, b"", b"")
        if ret != _NRT_SUCCESS:
            raise RuntimeError(f"nrt_init failed with code {ret}")
        _nrt_initialized = True


def nrt_close():
    """Shut down the Neuron Runtime."""
    global _nrt_initialized
    if _nrt_initialized:
        _libnrt.nrt_close()
        _nrt_initialized = False


class NrtModel:
    """Load a NEFF and expose it as a callable + benchmarkable object.

    Usage::

        model = NrtModel("/path/to/kernel.neff")
        output = model(input_a, input_b)       # correctness
        p99_ms = model.benchmark(warmup=10, iters=100)  # latency
        model.cleanup()
    """

    def __init__(self, neff_path: str):
        nrt_init()
        with open(neff_path, "rb") as f:
            self._neff_bytes = f.read()

        self._model = ctypes.c_void_p()
        ret = _libnrt.nrt_load(
            self._neff_bytes, len(self._neff_bytes),
            0, -1, ctypes.byref(self._model),
        )
        if ret != _NRT_SUCCESS:
            raise RuntimeError(f"nrt_load failed with code {ret}")

        # Discover tensor layout
        self._tensor_info_ptr = ctypes.c_void_p()
        ret = _libnrt.nrt_get_model_tensor_info(
            self._model, ctypes.byref(self._tensor_info_ptr),
        )
        if ret != _NRT_SUCCESS:
            raise RuntimeError(f"nrt_get_model_tensor_info failed: {ret}")

        count = ctypes.cast(
            self._tensor_info_ptr, ctypes.POINTER(ctypes.c_uint64),
        ).contents.value
        arr = ctypes.cast(
            ctypes.c_void_p(
                self._tensor_info_ptr.value + ctypes.sizeof(ctypes.c_uint64)
            ),
            ctypes.POINTER(_NrtTensorInfo),
        )

        self._inputs_info = []   # [(name_bytes, size, np_dtype, shape), ...]
        self._outputs_info = []
        for i in range(count):
            ti = arr[i]
            name = ti.name
            size = ti.size
            dtype = _NRT_DTYPE_MAP.get(ti.dtype, np.float32)
            shape = tuple(ti.shape[j] for j in range(ti.ndim)) if ti.ndim > 0 else (size // np.dtype(dtype).itemsize,)
            info = (name, size, dtype, shape)
            if ti.usage == _NRT_TENSOR_USAGE_INPUT:
                self._inputs_info.append(info)
            else:
                self._outputs_info.append(info)

        # Allocate tensor sets and tensors
        self._input_tset = ctypes.c_void_p()
        self._output_tset = ctypes.c_void_p()
        _libnrt.nrt_allocate_tensor_set(ctypes.byref(self._input_tset))
        _libnrt.nrt_allocate_tensor_set(ctypes.byref(self._output_tset))

        self._input_tensors = []
        for name, size, _, _ in self._inputs_info:
            t = ctypes.c_void_p()
            _libnrt.nrt_tensor_allocate(
                _NRT_TENSOR_PLACEMENT_DEVICE, 0, size, name, ctypes.byref(t),
            )
            _libnrt.nrt_add_tensor_to_tensor_set(self._input_tset, name, t)
            self._input_tensors.append(t)

        self._output_tensors = []
        for name, size, _, _ in self._outputs_info:
            t = ctypes.c_void_p()
            _libnrt.nrt_tensor_allocate(
                _NRT_TENSOR_PLACEMENT_DEVICE, 0, size, name, ctypes.byref(t),
            )
            _libnrt.nrt_add_tensor_to_tensor_set(self._output_tset, name, t)
            self._output_tensors.append(t)

        self._cleaned_up = False

    # ---- Callable interface (drop-in for @nki.jit / @nki.baremetal) ----

    def __call__(self, *args, **kwargs):
        """Execute the NEFF with the given numpy input arrays.

        Positional numpy arrays are matched to input tensors in order.
        Non-numpy args and all kwargs are ignored (compile-time constants
        already baked into the NEFF).
        """
        np_args = [a for a in args if isinstance(a, np.ndarray)]

        for i, (_, size, _, _) in enumerate(self._inputs_info):
            if i < len(np_args):
                arr = np.ascontiguousarray(np_args[i])
                _libnrt.nrt_tensor_write(
                    self._input_tensors[i], arr.ctypes.data, 0, size,
                )

        ret = _libnrt.nrt_execute(
            self._model, self._input_tset, self._output_tset,
        )
        if ret != _NRT_SUCCESS:
            raise RuntimeError(f"nrt_execute failed with code {ret}")

        results = []
        for i, (_, size, dtype, shape) in enumerate(self._outputs_info):
            buf = (ctypes.c_char * size)()
            _libnrt.nrt_tensor_read(self._output_tensors[i], buf, 0, size)
            arr = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
            results.append(arr)

        return results[0] if len(results) == 1 else tuple(results)

    # ---- Benchmarking ----

    def benchmark(self, warmup: int = 10, iters: int = 100) -> float:
        """Run nrt_execute in a loop and return P99 latency in milliseconds.

        Inputs from the most recent __call__ are reused.
        """
        for _ in range(warmup):
            _libnrt.nrt_execute(
                self._model, self._input_tset, self._output_tset,
            )

        latencies = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _libnrt.nrt_execute(
                self._model, self._input_tset, self._output_tset,
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        latencies.sort()
        return latencies[int(len(latencies) * 0.99)]

    # ---- Cleanup ----

    def cleanup(self):
        """Release all NRT resources for this model."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        for t in self._input_tensors + self._output_tensors:
            t_copy = ctypes.c_void_p(t.value)
            _libnrt.nrt_tensor_free(ctypes.byref(t_copy))

        tset = ctypes.c_void_p(self._input_tset.value)
        _libnrt.nrt_destroy_tensor_set(ctypes.byref(tset))
        tset = ctypes.c_void_p(self._output_tset.value)
        _libnrt.nrt_destroy_tensor_set(ctypes.byref(tset))

        _libnrt.nrt_free_model_tensor_info(self._tensor_info_ptr)
        _libnrt.nrt_unload(self._model)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
