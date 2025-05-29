import os
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

from analysis.run_analysis import run_analysis

if __name__ == "__main__":
    run_analysis() 