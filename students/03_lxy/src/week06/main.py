from pathlib import Path
import sys

# Ensure src is importable when running from week06 root
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from src.main import main

if __name__ == "__main__":
    main()
