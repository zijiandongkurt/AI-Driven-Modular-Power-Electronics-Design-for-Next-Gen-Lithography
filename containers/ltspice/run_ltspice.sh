#!/bin/bash
# Launch ONE LTspice batch simulation via an Apptainer container.
# Called by pipeline/simulation/ltspice_runner_snellius.py for each
# validated netlist.
#
# Env-var overrides (all optional):
#     LTSPICE_SIF        path to the .sif       default $HOME/container_custom.sif
#     LTSPICE_FILES_DIR  host dir to bind-mount default $HOME/ltspice-files
#                        (the Python runner stages .net files here, then
#                         they appear at /sim/ inside the container)
#     XVFB_RESOLUTION    Xvfb screen geometry   default 1024x768x16

IMAGE="${LTSPICE_SIF:-$HOME/container_custom.sif}"
LTSPICE_FILES_DIR="${LTSPICE_FILES_DIR:-$HOME/ltspice-files}"
WINE_PREFIX_DIR="/tmp/wine-prefix"
LTSPICE_PATH="${WINE_PREFIX_DIR}/drive_c/Program Files/ADI/LTspice/LTspice.exe"
XVFB_DISPLAY=":99"
XVFB_RESOLUTION="${XVFB_RESOLUTION:-1024x768x16}"
RUN_FILE=${1}

RUN_CMD='cp -a --no-preserve=ownership /opt/wineprefix-template /tmp/wine-prefix; \
             echo "copy success"; \
	     DISPLAY="${XVFB_DISP}" LD_PRELOAD= taskset -c 0-30 wine "$LTSPICE_EXE" -b 2>/dev/null || true; \
	     echo "wine init success"; \
             DISPLAY="${XVFB_DISP}" LD_PRELOAD= taskset -c 0-30 wineserver --wait 2>/dev/null || true; \
             echo "wineserver init success"; \
	     Xvfb "${XVFB_DISP}" -screen 0 "${XVFB_RES}" +extension GLX +render -nolisten tcp 2>/dev/null & \
             echo "xvfb success"; \
             export XVFB_PID=$!; \
             sleep 2; \
	     cd /sim; \
             DISPLAY="${XVFB_DISP}" LD_PRELOAD= taskset -c 0-30 wine "$LTSPICE_EXE" -b -run ${THE_FILE} || true; \
             kill $XVFB_PID || true; \
	     echo "container done"'

# Execute run
apptainer run --writable-tmpfs --cleanenv --fakeroot \
        --env LTSPICE_EXE="$LTSPICE_PATH" \
        --env XVFB_DISP="$XVFB_DISPLAY" \
        --env XVFB_RES="$XVFB_RESOLUTION" \
        --env THE_FILE="$RUN_FILE" \
        --bind "$LTSPICE_FILES_DIR":/sim \
        "$IMAGE" \
        /bin/bash -c "$RUN_CMD"

