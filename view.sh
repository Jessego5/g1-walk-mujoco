#!/bin/bash
# Live MuJoCo viewer for a trained policy (macOS needs mjpython; uv's Python needs its libpython on the path).
#   ./view.sh logs/g1/run1_full/model_2999.pt --vx 0.5 --seconds 60
cd "$(dirname "$0")"
PYLIB="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
DYLD_LIBRARY_PATH="$PYLIB" exec .venv/bin/mjpython play.py --viewer "$@"
