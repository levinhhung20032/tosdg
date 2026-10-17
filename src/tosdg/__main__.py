import sys

from .cli import main

if __name__ == "__main__":          # worker processes (parallel.py, spawn) import this module: they must not run the CLI
    sys.exit(main())
