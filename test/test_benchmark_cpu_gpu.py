import importlib.util
import unittest
import warnings
from pathlib import Path


_BENCHMARK_PATH = Path(__file__).with_name("benchmark_cpu_gpu.py")
_SPEC = importlib.util.spec_from_file_location(
    "storm_benchmark_cpu_gpu", _BENCHMARK_PATH
)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


class TestBenchmarkGpuFallback(unittest.TestCase):
    @staticmethod
    def _fallback_call():
        warnings.warn(
            "GPU computation failed; using CPU instead: synthetic failure",
            RuntimeWarning,
        )
        return "cpu result"

    def test_normal_calls_may_ignore_warnings(self):
        result = benchmark._run_call(
            self._fallback_call, reject_gpu_fallback=False
        )
        self.assertEqual(result, "cpu result")

    def test_gpu_benchmark_rejects_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            benchmark._run_call(
                self._fallback_call, reject_gpu_fallback=True
            )


if __name__ == "__main__":
    unittest.main()
