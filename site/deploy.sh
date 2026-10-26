#!/bin/sh
# Copies the QAL page to a web server directory (sudo on the host; SELinux context restored).
# Usage: site/deploy.sh <ssh-host> <directory>      live: site/deploy.sh e44 /var/www/qalang.de  (https://qalang.de)
set -eu
host="${1:?ssh host, e.g. e44}"
dir="${2:?target directory on the host, e.g. /var/www/qalang.de}"
here="$(cd "$(dirname "$0")" && pwd)"
scp -q "$here/index.html" "$host:/tmp/qal-index.html"
ssh "$host" "sudo mkdir -p '$dir' && sudo install -m 644 /tmp/qal-index.html '$dir/index.html' && rm /tmp/qal-index.html && (sudo restorecon -R '$dir' 2>/dev/null || true)"
echo "deployed to $host:$dir/index.html"
