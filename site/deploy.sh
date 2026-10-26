#!/bin/sh
# Copies the QAL page to a web server directory.
# Usage: site/deploy.sh <ssh-host> <directory>      e.g. site/deploy.sh e44 /var/www/qal
set -eu
host="${1:?ssh host, e.g. e44}"
dir="${2:?target directory on the host, e.g. /var/www/qal}"
here="$(cd "$(dirname "$0")" && pwd)"
ssh "$host" "mkdir -p '$dir'"
scp "$here/index.html" "$host:$dir/index.html"
echo "deployed to $host:$dir/index.html"
