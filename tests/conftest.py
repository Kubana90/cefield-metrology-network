"""
pytest configuration — adds cloud-core/ to PYTHONPATH
so all test modules can import baseline, normalizer, etc. directly
without sys.path manipulation inside individual test files.
"""
import sys
import os

# Resolve cloud-core absolute path regardless of where pytest is invoked from
_CLOUD_CORE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cloud-core")
)
if _CLOUD_CORE not in sys.path:
    sys.path.insert(0, _CLOUD_CORE)
