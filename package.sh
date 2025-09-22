#!/usr/bin/env bash
set -euo pipefail

ROOT="MyDecryptor"
ZIPNAME="MyDecryptor.zip"

if [ ! -d "$ROOT" ]; then
  echo "Project root '$ROOT' not found."
  exit 1
fi

pushd "$ROOT" >/dev/null
make clean || true
popd >/dev/null

rm -f "$ZIPNAME"
zip -r "$ZIPNAME" "$ROOT" \
  -x "*.o" \
  -x "$ROOT/Decryptor" \
  -x "__pycache__/*" \
  -x ".git/*"

echo "Packaged into $ZIPNAME"