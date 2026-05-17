#!/bin/bash
# =====================================================================
# build.sh — get the LTspice + Wine Apptainer container
# =====================================================================
# Two modes (pick by setting MODE env var):
#
#   MODE=pull    (default)  Pull aanas0sayed/docker-ltspice from Docker
#                Hub (~3 minutes, no fakeroot).
#
#   MODE=build              Build from Apptainer.def locally
#                (~15-20 minutes, needs fakeroot or remote).
#
# Usage:
#     bash containers/ltspice/build.sh                            # mode=pull, target=$HOME/container_custom.sif
#     bash containers/ltspice/build.sh /scratch/$USER/lts.sif     # custom target, still pull
#     MODE=build bash containers/ltspice/build.sh                 # build from .def instead
#
# Both modes produce the same on-disk layout the team's
# run_ltspice.sh / run_ltspice_snellius.sh expect.
# =====================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF="$SCRIPT_DIR/Apptainer.def"
SIF="${1:-$HOME/container_custom.sif}"
MODE="${MODE:-pull}"

# ── Sanity ──────────────────────────────────────────────────────────
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found in PATH." >&2
    echo "  On Snellius:  module load 2023 && module load Apptainer/1.2.5-GCCcore-12.3.0" >&2
    echo "  (exact version may differ — \`module spider Apptainer\` to list)" >&2
    exit 1
fi

echo "======================================================================"
echo " LTspice container provisioning"
echo "======================================================================"
echo "  Mode   : $MODE  (pull = Docker Hub, build = local --fakeroot)"
echo "  Target : $SIF"
echo "======================================================================"

case "$MODE" in
  pull)
    # ── Pull pre-built image from Docker Hub (fast, ~3 min) ──────────
    if [ -f "$SIF" ]; then
        echo "WARNING: $SIF already exists; will overwrite."
        rm -f "$SIF"
    fi
    apptainer pull "$SIF" docker://aanas0sayed/docker-ltspice
    ;;

  build)
    # ── Build from .def (slower, ~15-20 min, but fully reproducible) ─
    if [ ! -f "$DEF" ]; then
        echo "ERROR: $DEF not found." >&2
        exit 2
    fi
    if apptainer build --fakeroot "$SIF" "$DEF" 2>&1 | tee /tmp/apptainer_build.log; then
        BUILD_OK=1
    else
        BUILD_OK=0
    fi
    if [ "$BUILD_OK" != "1" ]; then
        echo ""
        echo "----------------------------------------------------------------------"
        echo " Local --fakeroot build failed.  Trying --remote (Sylabs)."
        echo " If you haven't logged in yet:"
        echo "     apptainer remote login                              # interactive"
        echo "     # or  apptainer remote login --tokenfile <token>    # CI"
        echo "     # token from https://cloud.sylabs.io/auth/tokens"
        echo "----------------------------------------------------------------------"
        apptainer build --remote "$SIF" "$DEF"
    fi
    ;;

  *)
    echo "ERROR: unknown MODE='$MODE'.  Use MODE=pull or MODE=build" >&2
    exit 3
    ;;
esac

# ── Verify ──────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " Build complete.  Verifying container ..."
echo "======================================================================"
ls -lh "$SIF"

# Quick functional test
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/smoke.net" <<'EOF'
* smoke-test RC: V1->R1->C1->GND with .tran
V1 in 0 5
R1 in out 1k
C1 out 0 1u
.tran 1u 5m
.end
EOF

echo "Running smoke test (RC circuit through container) ..."
if apptainer run --bind "$TMPDIR":/sim "$SIF" /sim/smoke.net 2>&1 | tail -10; then
    if [ -f "$TMPDIR/smoke.raw" ]; then
        echo "✅ smoke test PASS  ($(stat -c%s "$TMPDIR/smoke.raw") bytes of .raw)"
    else
        echo "⚠️  Container ran but no .raw produced"
    fi
else
    echo "❌ smoke test FAIL"
fi
rm -rf "$TMPDIR"

echo ""
echo "Next:"
echo "  export LTSPICE_SIF=$SIF"
echo "  export LTSPICE_LAUNCHER=$SCRIPT_DIR/run_ltspice_snellius.sh   # Snellius"
echo "  ##export LTSPICE_LAUNCHER=$SCRIPT_DIR/run_ltspice.sh           # workstation"
echo "  export LTSPICE_FILES_DIR=\$HOME/ltspice-files"
echo "  mkdir -p \$LTSPICE_FILES_DIR"
echo "  bash scripts/run_pipeline.sh full"
