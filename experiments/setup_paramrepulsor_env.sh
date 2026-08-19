#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
python="${PARAMREPULSOR_PYTHON:-/home/enjalot/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10}"
lock="$repo_root/requirements/paramrepulsor-cu124.lock.txt"
venv="$repo_root/.venv"

if [[ ! -x "$python" ]]; then
  printf 'Python 3.10 is absent at %s\n' "$python" >&2
  exit 1
fi
if [[ ! -f "$lock" ]]; then
  printf 'ParamRepulsor lock is absent at %s\n' "$lock" >&2
  exit 1
fi
if [[ -e "$venv" ]]; then
  printf 'Refusing existing environment at %s\n' "$venv" >&2
  exit 1
fi

uv venv --python "$python" "$venv"
uv pip sync --python "$venv/bin/python" "$lock"
printf 'Prepared pinned ParamRepulsor environment at %s\n' "$venv"
