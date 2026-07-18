"""The one controller-admitted executable for all six Round 0015 nodes."""
from __future__ import annotations

from experiments.run_round0014_node import configure_round0015, main


if __name__ == "__main__":
    configure_round0015()
    raise SystemExit(main())
