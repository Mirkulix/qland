#!/bin/sh
# Copies the QAL pages (German at /, English at /en/) and the compiler the playground runs in the
# browser to a web server directory (sudo on the host; SELinux context restored).
# Usage: site/deploy.sh <ssh-host> <directory>      live: site/deploy.sh e44 /var/www/qalang.de  (https://qalang.de)
# Rebuild the pages first with site/build.py.
set -eu
host="${1:?ssh host, e.g. e44}"
dir="${2:?target directory on the host, e.g. /var/www/qalang.de}"
here="$(cd "$(dirname "$0")" && pwd)"
scp -q "$here/index.html" "$host:/tmp/qal-index.html"
scp -q "$here/en/index.html" "$host:/tmp/qal-index-en.html"
# The compiler the playground runs: the checked-in seed, so the page always runs the current one.
scp -q "$here/../qal/selfhost/qalc.wasm" "$host:/tmp/qal-qalc.wasm"
ssh "$host" "sudo mkdir -p '$dir/en' && sudo install -m 644 /tmp/qal-index.html '$dir/index.html' && sudo install -m 644 /tmp/qal-index-en.html '$dir/en/index.html' && sudo install -m 644 /tmp/qal-qalc.wasm '$dir/qalc.wasm' && rm /tmp/qal-index.html /tmp/qal-index-en.html /tmp/qal-qalc.wasm && (sudo restorecon -R '$dir' 2>/dev/null || true)"
echo "deployed to $host:$dir (index.html, en/index.html, qalc.wasm)"
