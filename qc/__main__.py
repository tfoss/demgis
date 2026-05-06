"""Entry point so `python -m qc <subcommand>` works."""
import sys
from qc.cli import main

sys.exit(main())
