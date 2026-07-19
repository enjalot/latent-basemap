"""The six-node Round 0019 duplicate-multiplicity treatment executable."""
from __future__ import annotations

from experiments.run_round0014_node import configure_round0019, main


if __name__ == "__main__":
    configure_round0019()
    raise SystemExit(main())
