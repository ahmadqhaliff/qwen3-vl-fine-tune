#!/usr/bin/env bash
set -euo pipefail

SRC="/root/qwen3-vl-fine-tune"
DST="/workspace/qwen3-vl-fine-tune"

mkdir -p "${DST}"

if [ -d "${SRC}" ]; then
  echo "Syncing ${SRC} -> ${DST} (repo + data + outputs)..."
  rsync -a --info=progress2 "${SRC}/" "${DST}/"
  echo "Keeping backup at ${SRC}.bak (if possible) and symlinking ${SRC} -> ${DST}"
  mv "${SRC}" "${SRC}.bak" 2>/dev/null || true
  ln -s "${DST}" "${SRC}" 2>/dev/null || true
else
  echo "${SRC} not found; nothing to migrate."
  echo "If you cloned elsewhere, sync that folder into ${DST}."
fi

echo "Done. Work from: ${DST}"
