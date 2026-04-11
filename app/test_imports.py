"""Quick test to verify all imports work."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from core import PATHS
    print("✅ core/__init__.py imports OK")
    print(f"   PROJECT_ROOT: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
except Exception as e:
    print(f"❌ core/__init__.py failed: {e}")

try:
    from core.data_loader import load_dataset
    print("✅ core/data_loader.py imports OK")
except Exception as e:
    print(f"❌ core/data_loader.py failed: {e}")

try:
    from core.metrics import calculate_all_metrics
    print("✅ core/metrics.py imports OK")
except Exception as e:
    print(f"❌ core/metrics.py failed: {e}")

try:
    from core.fusion import run_fusion
    print("✅ core/fusion.py imports OK")
except Exception as e:
    print(f"❌ core/fusion.py failed: {e}")

print("\nAll imports successful!")