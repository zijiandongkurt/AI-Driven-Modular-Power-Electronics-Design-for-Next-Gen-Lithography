# LTspice + Wine Container — Team Edition

Headless LTspice via Apptainer/Singularity (or Podman/Docker).
**All files in this folder are publicly buildable — no team-private artifacts required.**

```
containers/ltspice/
├── Apptainer.def              ← team-tested Debian + Wine 11 + LTspice 64
├── build.sh                   ← one-shot builder (calls apptainer build)
├── run_ltspice.sh             ← per-simulation launcher (workstation)
├── run_ltspice_snellius.sh    ← same, with --no-mount hostfs for Snellius
├── podman_commands.txt        ← Docker Hub pre-built image alternative
└── README.md                  ← this file
```

---

## Three ways to get the container

### Option A — pull pre-built Docker image (fastest, ~3 min)

There's a public Docker Hub image **`aanas0sayed/docker-ltspice`** that
the team already uses for cross-checking.  Pull it once into an
Apptainer .sif:

```bash
module load 2023
module load Apptainer/1.2.5-GCCcore-12.3.0     # adjust version via `module spider Apptainer`

# Pull from Docker Hub directly to a .sif
apptainer pull $HOME/container_custom.sif docker://aanas0sayed/docker-ltspice
```

**Pros**: ~3 min, no build, no fakeroot needed.
**Cons**: third-party image (the team uses it but you can't audit Wine version etc.).

### Option B — build from `Apptainer.def` (10–20 min)

Reproducible, you own every layer:

```bash
module load 2023
module load Apptainer/1.2.5-GCCcore-12.3.0

bash containers/ltspice/build.sh
# or with a custom target path:
bash containers/ltspice/build.sh $HOME/container_custom.sif
```

The build script tries `--fakeroot` first, then `--remote` (Sylabs build server).

**Pros**: official versions pinned (Wine 11.0.0.0~bookworm-1 + LTspice 64 MSI from AdI).
**Cons**: 10–20 min; needs `--fakeroot` or remote Sylabs account.

### Option C — Docker / Podman (non-HPC desktop)

See `podman_commands.txt` for the exact one-liner used for development:

```bash
podman run \
    --mount type=bind,source=/path/to/your/netlists,destination=/sim \
    aanas0sayed/docker-ltspice \
    ltspice -b -run "Z:\\sim\\my-netlist.asc"
```

(Useful for testing the container itself outside HPC.)

---

## Container internals (what `Apptainer.def` actually does)

```
1. Bootstrap from  debian:bookworm
2. apt install     ca-certificates, wget, locales, xvfb, cabextract,
                   winbind, gosu, p7zip-full, unzip, libvulkan1
3. dpkg --add-architecture i386       ← CRUCIAL for Wine
4. add WineHQ apt key + repo, install pinned Wine 11.0.0.0~bookworm-1
   (winehq-stable + amd64 + i386 + dev packages)
5. install winetricks
6. start Xvfb at :99 (headless X display)
7. winetricks riched20 riched30         ← LTspice's rich-edit dependency
8. wineboot --init                       ← initialize Wine prefix
9. wget LTspice64.msi from
       https://ltspice.analog.com/software/LTspice64.msi
10. wine msiexec /i LTspice64.msi /quiet /norestart   ← silent install
11. relocate install:
       /home/wineuser/.wine/drive_c/Program Files/ADI/LTspice → /opt/ltspice
       /home/wineuser/.wine                                   → /opt/wineprefix-template
12. lock permissions (read-only, world-readable)
```

At runtime (`run_ltspice.sh`):
```
1. apptainer run --writable-tmpfs ...
2. inside container:  copy /opt/wineprefix-template → /tmp/wine-prefix
                      (so each run gets a fresh writable prefix)
3. start Xvfb on :99
4. cd /sim (= host's $LTSPICE_FILES_DIR bind-mounted)
5. wine LTspice.exe -b -run topX.net
6. .raw lands at /sim/topX.raw → visible on host as $LTSPICE_FILES_DIR/topX.raw
7. kill Xvfb
```

---

## Wire it up to our pipeline

Once the .sif is built/pulled:

```bash
mkdir -p $HOME/ltspice-files

# Point the Python LTspice runner at our scripts (env-var overrides):
export LTSPICE_SIF=$HOME/container_custom.sif
export LTSPICE_LAUNCHER=$PWD/containers/ltspice/run_ltspice_snellius.sh   # Snellius
# or workstation:
##export LTSPICE_LAUNCHER=$PWD/containers/ltspice/run_ltspice.sh
export LTSPICE_FILES_DIR=$HOME/ltspice-files

# Smoke test the container directly (without our Python pipeline):
cp pipeline/data/batch_sft_expanded_idx0/llm_output/top1.net $HOME/ltspice-files/
bash $LTSPICE_LAUNCHER /sim/top1.net
ls -lh $HOME/ltspice-files/top1.raw      # should appear

# If above worked, run the full GRPO loop:
bash scripts/run_pipeline.sh full
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `apptainer: command not found` | module not loaded | `module spider Apptainer` then `module load <version>` |
| `FATAL: --fakeroot requires user namespaces` | unprivileged builds disabled on this HPC | use `apptainer build --remote` (needs Sylabs login: `apptainer remote login`) or Option A (pre-built pull) |
| build fails at `wget ... LTspice64.msi` (404) | AdI changed the URL | edit `Apptainer.def` → try `https://ltspice.analog.com/software/LTspiceXVII.exe` (legacy) |
| build fails at `apt install winehq-stable=11.0.0.0~bookworm-1` | Wine version moved | edit `Apptainer.def` → bump `WINE_VERSION=` to whatever is current (`apt-cache madison winehq-stable` inside a debian container) |
| build fails at `wine msiexec /i ...` | Wine's rich-edit deps missing | confirm winetricks ran: container has `riched20` + `riched30` |
| `run_ltspice.sh` exits with "container done" but no `.raw` | LTspice quit before sim | check `/sim/topX.log` inside the container's bind dir for diagnostic |
| `aanas0sayed/docker-ltspice` rate-limited | Docker Hub anonymous pull limit | log in: `apptainer registry login docker.io` (uses a Docker Hub token) |
| Python runner errors: "Run script not found" | env vars not exported | `echo $LTSPICE_LAUNCHER` ; re-export, re-run |
| OOM during forward at end of step | LTspice container eats RAM | kill stale `wineserver` processes: `pkill -9 wineserver` |

---

## Disk + memory

| Resource | Pre-built pull (A) | Build from def (B) |
|---|---|---|
| .sif size | ~1.2 GB | ~1.5 GB |
| Build time | ~3 min (network) | ~15-20 min (network + apt + msi install) |
| Build RAM | < 1 GB | < 2 GB |
| Per-sim RAM | ~400-800 MB (Wine + LTspice) | same |
| Per-sim time | ~10-30 s for a 5ms buck transient | same |
