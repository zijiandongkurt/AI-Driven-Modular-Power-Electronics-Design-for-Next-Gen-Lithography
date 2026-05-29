#!/bin/bash

IMAGE=$HOME/container_custom.sif
WINE_PREFIX_DIR="/tmp/wine-prefix"
LTSPICE_PATH="${WINE_PREFIX_DIR}/drive_c/Program Files/ADI/LTspice/LTspice.exe"
XVFB_DISPLAY=":99"
XVFB_RESOLUTION="${XVFB_RESOLUTION:-1024x768x16}"

# FIX: Convert the Linux path (/sim/filename) to a Windows path (Z:\sim\filename)
FILENAME=$(basename "${1}")
RUN_FILE="Z:\\sim\\${FILENAME}"

RUN_CMD='cp -a --no-preserve=ownership /opt/wineprefix-template /tmp/wine-prefix; \
             echo "copy success"; \
             Xvfb "${XVFB_DISP}" -screen 0 "${XVFB_RES}" +extension GLX +render -nolisten tcp 2>/dev/null & \
             export XVFB_PID=$!; \
             echo "xvfb success"; \
             sleep 2; \
             cd /sim; \
             DISPLAY="${XVFB_DISP}" LD_PRELOAD= wine "$LTSPICE_EXE" -b -run "${THE_FILE}" || true; \
             kill $XVFB_PID || true; \
             wineserver -k || true; \
             echo "container done"'

# Execute run
apptainer run --writable-tmpfs --cleanenv --no-mount hostfs --fakeroot --env LTSPICE_EXE="$LTSPICE_PATH" --env XVFB_DISP="$XVFB_DISPLAY" --env XVFB_RES="$XVFB_RESOLUTION" --env THE_FILE="$RUN_FILE" \
        --bind $HOME/ltspice-files/:/sim \
        "$IMAGE" \
        /bin/bash -c "$RUN_CMD"