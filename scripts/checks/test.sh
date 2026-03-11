#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
	ROOT_DIR="$ROOT_DIR_GIT"
else
	ROOT_DIR="$SCRIPT_DIR"
fi


echo ROOT_DIR: "$ROOT_DIR"