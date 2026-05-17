#!/bin/bash
# =====================================================================
# run_ltspice_docker.sh — launcher for the pulled `aanas0sayed/docker-ltspice` image
# =====================================================================
# Use this when your $LTSPICE_SIF was created via:
#       apptainer pull <sif> docker://aanas0sayed/docker-ltspice
# (i.e. the default MODE=pull path in build.sh).
#
# Compared to run_ltspice.sh / run_ltspice_snellius.sh (which target
# OUR Apptainer.def build), this script does NOT try to copy the
# /opt/wineprefix-template directory — that path only exists in the
# def-built image.  The aanas0sayed image has its own entrypoint that
# primes Wine + Xvfb internally, and exposes a `ltspice` wrapper that
# we just hand the .net file to.
#
# Env-var overrides:
#     LTSPICE_SIF        default $HOME/container_custom.sif
#     LTSPICE_FILES_DIR  default $HOME/ltspice-files
#
# Usage:
#     bash run_ltspice_docker.sh /sim/<netfile.net>
# =====================================================================

IMAGE="${LTSPICE_SIF:-$HOME/container_custom.sif}"
LTSPICE_FILES_DIR="${LTSPICE_FILES_DIR:-$HOME/ltspice-files}"
RUN_FILE="${1:?Usage: $0 /sim/<netfile.net>}"

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: container not found at $IMAGE" >&2
    echo "  Pull it: apptainer pull $IMAGE docker://aanas0sayed/docker-ltspice" >&2
    exit 2
fi
if [ ! -d "$LTSPICE_FILES_DIR" ]; then
    echo "ERROR: bind-mount source missing: $LTSPICE_FILES_DIR" >&2
    echo "  mkdir -p $LTSPICE_FILES_DIR" >&2
    exit 3
fi

# Convert "/sim/topX.net" to "Z:\sim\topX.net" — the Wine convention:
# drive Z: maps to / inside the container, so /sim/foo.net → Z:\sim\foo.net
NETBASE=$(basename "$RUN_FILE")
WINE_PATH='Z:\sim\'"$NETBASE"

# Call the image's entrypoint with `ltspice -b -run <winepath>`.
# The entrypoint handles wineboot + xvfb internally before exec-ing this.
# --no-mount hostfs is needed on Snellius (HPC quirk); harmless on workstations.
apptainer run --writable-tmpfs --cleanenv --no-mount hostfs --fakeroot \
        --bind "$LTSPICE_FILES_DIR":/sim \
        "$IMAGE" \
        ltspice -b -run "$WINE_PATH"

RC=$?
# Confirm a .raw was produced (entrypoint may exit 0 even if LTspice quit silently)
RAW_PATH="$LTSPICE_FILES_DIR/${NETBASE%.net}.raw"
if [ -f "$RAW_PATH" ]; then
    echo "OK: $RAW_PATH ($(stat -c%s "$RAW_PATH") bytes)"
    exit 0
else
    echo "WARN: no .raw produced at $RAW_PATH (LTspice may have failed)" >&2
    exit ${RC:-1}
fi
