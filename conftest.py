import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.X import Y` works
# in both pytest and the IDE without installing the package.
sys.path.insert(0, str(Path(__file__).parent))
