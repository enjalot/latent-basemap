"""The one controller-admitted executable for all six Round 0017 nodes."""
from __future__ import annotations

from experiments.run_round0014_node import configure_round0017, main


if __name__ == "__main__":
    configure_round0017()
    raise SystemExit(main())
