#!/bin/sh
set -eu
ulimit -c 0
if [ "$#" -eq 0 ]; then set -- ./scratch/test_desktop; fi
task_dir=$(mktemp -d /tmp/paraketto-desktop-test.XXXXXX)
trap 'rm -rf "$task_dir"' EXIT HUP INT TERM
mkdir -m 700 "$task_dir/runtime" "$task_dir/config" "$task_dir/cache" "$task_dir/data"
# No desktop settings, IBus engine, or typed text escape this private session.
env -u IBUS_ADDRESS -u DBUS_SESSION_BUS_ADDRESS \
    XDG_RUNTIME_DIR="$task_dir/runtime" XDG_CONFIG_HOME="$task_dir/config" XDG_CACHE_HOME="$task_dir/cache" \
    XDG_DATA_HOME="$task_dir/data" \
    GTK_IM_MODULE=gtk-im-context-simple GSETTINGS_BACKEND=memory GIO_USE_VFS=local NO_AT_BRIDGE=1 \
    dbus-run-session --config-file tests/desktop-session.conf -- xvfb-run -a sh -eu -c '
        export IBUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/ibus"
        ibus-daemon --daemonize --xim --panel disable --config disable --emoji-extension disable --cache none --address "$IBUS_ADDRESS"
        trap "ibus exit >/dev/null 2>&1 || true" EXIT
        tries=0
        until ibus engine xkb:us::eng >"$XDG_RUNTIME_DIR/ibus-start.log" 2>&1; do
            tries=$((tries + 1))
            if [ "$tries" -ge 40 ]; then cat "$XDG_RUNTIME_DIR/ibus-start.log"; exit 1; fi
            sleep 0.05
        done
        "$@"
    ' paraketto-tests "$@"
