"""
Root entry point for random_split.py
Executes the random split pipeline from src/random_split.py.
"""
import sys
import os

# Add src/ to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from random_split import main

if __name__ == "__main__":
    main()
