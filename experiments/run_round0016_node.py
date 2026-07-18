"""The one controller-admitted executable for all six Round 0016 nodes."""
from __future__ import annotations

from experiments.run_round0014_node import configure_round0016, main


if __name__ == "__main__":
    configure_round0016()
    raise SystemExit(main())
