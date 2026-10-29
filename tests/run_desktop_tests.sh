#!/bin/sh
set -eu
ulimit -c 0
if [ "$#" -eq 0 ]; then set -- ./scratch/test_desktop; fi
task_dir=$(mktemp -d /tmp/paraketto-desktop-test.XXXXXX)
trap 'rm -rf "$task_dir"' EXIT HUP INT TERM
mkdir -m 700 "$task_dir/runtime" "$task_dir/config" "$task_dir/cache" "$task_dir/data"
# No desktop settings, clipboard contents, or typed text escape this private session.
env -u IBUS_ADDRESS -u DBUS_SESSION_BUS_ADDRESS \
    XDG_RUNTIME_DIR="$task_dir/runtime" XDG_CONFIG_HOME="$task_dir/config" XDG_CACHE_HOME="$task_dir/cache" \
    XDG_DATA_HOME="$task_dir/data" \
    GTK_IM_MODULE=gtk-im-context-simple GSETTINGS_BACKEND=memory GIO_USE_VFS=local NO_AT_BRIDGE=1 \
    dbus-run-session --config-file tests/desktop-session.conf -- xvfb-run -a "$@"
