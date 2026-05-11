#!/usr/bin/env python3
"""
Main entry point for 03_lxy's regression analysis project.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.week07.main import main


if __name__ == "__main__":
    main()
