# Running the regional map on Sherlock — step-by-step

**What this does:** runs the *inference* half of [PLAN_RegionalMap.md](PLAN_RegionalMap.md)
on Stanford's Sherlock cluster — CTX imagery → Fang-ViT embed → trained head → calibrate →
**prediction GeoTIFFs** for the 7-tile circum-Chryse block — on a GPU. It does **not**
rebuild the cohort dataset (that already exists on your laptop). You run the steps; you copy
the small GeoTIFFs back; all the figures/validation stay local in `24_regional_map.ipynb`.

**Your Sherlock coordinates:** group `mlapotre`, home `/home/groups/mlapotre/bamaro`, GPU
partition `gpu`. Edit paths below if they change.

---

## The 10-minute mental model (read once)

Sherlock is a shared Linux cluster. Four things to internalize:

1. **Login node vs compute node.** When you connect you land on a *login node* — fine for
   editing files and submitting jobs, **not** for running the model. Real work runs on a
   *compute node*, which you get either **interactively** (a live terminal on a compute
   node) or by **submitting a batch job** (`sbatch`, runs unattended).
2. **Slurm** is the scheduler. You *request* resources (a GPU, CPUs, memory, a time limit)
   and it places your job on a compute node when one is free. You may wait in a queue.
3. **Storage tiers.** `$HOME` (15 GB, backed up) — the git repo. `$GROUP_HOME`
   (`/home/groups/mlapotre`, 1 TB, backed up) — the Python environment. `$SCRATCH` (100 TB,
   fast, **not** backed up, **purged after 90 days idle**) — the heavy CTX cache and the
   output GeoTIFFs. Rule: code/env on the backed-up tiers, big I/O on `$SCRATCH`.
4. **GPU.** The embedding step wants a GPU; everything else is trivial. You request one with
   partition `gpu` (interactively or in the batch script).

There are **two ways to drive Sherlock**. Both reach the same shell and run the same
commands — pick one:

- **Path A — OnDemand (web browser).** No SSH setup, drag-and-drop file upload/download,
  a terminal and JupyterLab in the browser. **Recommended for you.** → Part A.
- **Path B — SSH terminal.** Classic `ssh` + `scp`/`rsync`. → Part B.

After the setup path you chose, everyone does the **same Run + Retrieve** steps (Part C/D).

---

## Part A — OnDemand (web browser path)

OnDemand is Sherlock's web portal: <https://ondemand.sherlock.stanford.edu> (log in with
SUNet ID + the usual two-step). The Dashboard has menus for **Files**, **Clusters** (shell),
and **Interactive Apps** (JupyterLab, etc.).

**Order of operations** (A-steps are web-UI actions, Setup/Part-C steps run in the terminal —
they interleave, so follow this sequence, not the section numbers):

1. **A1** — launch JupyterLab on a GPU node, open a Terminal.
2. **Setup step 0** — `git clone` + `git checkout` (in that terminal). *Do this before A2.*
3. **A2** — upload `h2c_artifacts.tgz` via the Files browser into the just-cloned folder,
   then extract it in the terminal.
4. **Setup steps 1–3** — build the venv → symlink + fetch CTX tiles → parity gate.
5. **Part C** — throughput probe → `sbatch` the full run → monitor.
6. **A4 / Part D** — download the GeoTIFFs back to the laptop.

### A1. Get a terminal on a GPU compute node
This single session gives you a shell *and* a file browser for all of setup + the parity
check + the throughput probe.

- **Interactive Apps → JupyterLab.** Fill the form (labels vary slightly by OnDemand
  version) roughly as below, then **Launch**, wait in the queue, and **Connect to
  JupyterLab**:

  | Field | Choose | Why |
  |---|---|---|
  | **Partition** | `gpu` | the GPU partition |
  | **Number of GPUs** | **1** | ⚠️ the field that matters — without it `cuda_available` is `False` and everything runs on CPU |
  | CPUs | 8 | |
  | Memory | 32 GB | embedding windows fit comfortably |
  | Time | 3 hours | covers setup + parity + the probe |
  | **Python version** | `python/3.12.1` (or newest ≥ 3.10 offered) | matches `setup_sherlock_env.sh`; **not critical** — we activate our own venv in the terminal |
  | **Additional modules** | *leave empty* | the pip `torch` wheel bundles its own CUDA runtime (no `cuda/...` module); the venv has rasterio/numpy/jupyter/etc. |
  | **Workspace / working dir** | `$HOME/hirise2ctx` (or leave default and `cd` later) | just the starting folder; doesn't affect compute |

  **The JupyterLab app is only a vehicle to get a Terminal on a GPU node** — we `source` the
  venv for every command, so the form's Python/modules don't drive the real work. The one
  choice that affects correctness is **GPUs = 1 on `gpu`**.

- In JupyterLab open **File → New → Terminal**. That terminal is running **on a GPU node** —
  exactly where the setup and parity check should run.

  *(Alternative for a pure shell with no Jupyter: Dashboard → **Clusters → Sherlock Shell
  Access** gives you a login-node shell. Good for `sbatch`/`git`, but it is NOT a GPU node,
  so don't run the parity check there — use the JupyterLab terminal for anything that needs
  the GPU.)*

  *(If the app refuses to start because the chosen Python lacks Jupyter: our venv ships
  Jupyter via the `dev` extra, so after step 1 you can point the app's "custom environment"
  at `/home/groups/mlapotre/bamaro/envs/hirise2ctx`, or just do the non-GPU steps in a
  Sherlock Shell and the GPU-only parity check via `sh_dev -p gpu -G 1`.)*

### A2. Upload the trained artifacts via the Files browser
These three live only on your laptop and are gitignored (so `git clone` won't bring them).
On the **laptop**, bundle them into one file first. Run this from the repo root **on one
line** — it works in both PowerShell and Git Bash (Windows ships `tar.exe`). Do *not* use
bash-style `\` line-continuations in PowerShell; they aren't continuations there and tar
will error with `Couldn't visit directory`:
```
tar -czf h2c_artifacts.tgz models/deployable/86c51a5dca220f63 models/deployable/calibration.npz models/deployable/parity_ref.npz
```
Verify the bundle with `tar -tzf h2c_artifacts.tgz` (should list the head dir + the two
`.npz`). It's ~2.4 MB and safe to delete after upload.

**Do this after Setup step 0 (the clone), so the `hirise2ctx/` folder exists.** In OnDemand:
**Files → Home Directory** (`/home/users/bamaro`), navigate into `hirise2ctx/`, click
**Upload**, drop `h2c_artifacts.tgz`. Then in the terminal:
```bash
cd $HOME/hirise2ctx && tar -xzf h2c_artifacts.tgz && rm h2c_artifacts.tgz
ls models/deployable/      # should show 86c51a5dca220f63/  calibration.npz  parity_ref.npz
```

### A3. Now do the shared setup
In the JupyterLab terminal run **Setup steps 0–3** below, with the A2 upload slotted in right
after step 0 (per the order-of-operations list above).

### A4. Download the results when done
After the run, **Files → navigate to** `$SCRATCH/hirise2ctx/map_region` (the Files browser
can open `$SCRATCH`), select the `*_prob.tif` / `*_abundance.tif`, click **Download**. Drop
them into `reports/map_region/` on the laptop.

---

## Part B — SSH terminal path (alternative)

```bash
ssh <SUNetID>@login.sherlock.stanford.edu          # login node
```
Upload the artifacts from the **laptop** (after the clone in step 0):
```bash
rsync -avP models/deployable/86c51a5dca220f63 models/deployable/calibration.npz \
    models/deployable/parity_ref.npz \
    <SUNetID>@dtn.sherlock.stanford.edu:hirise2ctx/models/deployable/
```
For anything that needs the GPU (the parity check), drop onto a GPU compute node first:
```bash
sh_dev -p gpu -G 1 -c 8 -m 32GB -t 02:00:00        # interactive GPU node
```
Then do **Setup steps 0–3** below. Download results at the end with `rsync` (Part D).

---

## Setup (steps 0–3, same for both paths)

> Run these in a terminal that is on a **GPU compute node** (OnDemand JupyterLab terminal, or
> `sh_dev -p gpu -G 1`). Downloads and `git`/`sbatch` are fine anywhere; only the env smoke
> test and the parity check actually need the GPU.

> **Sherlock TLS escape hatch.** murray-lab.caltech.edu and Zenodo send incomplete cert
> chains, and Linux OpenSSL won't auto-fetch the missing intermediate (Windows does, so it
> works on the laptop but fails here with `CERTIFICATE_VERIFY_FAILED`). For these public,
> fixed-URL downloads only, set this **once per shell** before steps 1–2:
> ```bash
> export HIRISE2CTX_INSECURE_TLS=1
> ```
> It makes the checkpoint `curl` (step 1) and the Python tile fetch (step 2) skip cert
> verification (off by default; only honored when set). Not needed after the data is cached —
> the `sbatch` run reads from disk and does no downloads.

### 0. Get the code (and the right branch)
`git clone` checks out `main`, but this work is on a feature branch — check it out:
```bash
cd $HOME
git clone https://github.com/brianvamaro/hirise2ctx.git
cd hirise2ctx
git checkout fm-deployable-head-and-map-pilot      # the branch with map_region.py etc.
```
*(If you merge this branch to `main` first, you can skip the checkout.)* Then upload the
artifacts (A2 or B) so `models/deployable/` is populated.

### 1. Build the Python environment (no conda)
```bash
export HIRISE2CTX_INSECURE_TLS=1     # see the TLS note above; needed for the checkpoint download
bash setup_sherlock_env.sh
```
What it does, and what to expect: loads `python/3.12.1` (verifying it provides a venv-capable
`python3`), creates a venv at `/home/groups/mlapotre/bamaro/envs/hirise2ctx` (backed up),
installs **CUDA** PyTorch + the project deps **as wheels only** (`--only-binary`, so Sherlock's
older glibc gets compatible wheels instead of failing source builds), and downloads the
341 MB Fang-ViT checkpoint from Zenodo into `models/pretrained/` (with the TLS escape hatch
above). It ends by importing the full inference chain and printing
`inference imports OK … device NVIDIA L40S` — if `cuda_available` is `False`, you launched on
a non-GPU node (relaunch the app with Partition = `gpu`, GPUs = 1). First run takes a few
minutes (downloads wheels + the checkpoint; compiles nothing).

> `truststore` (Windows) and `certifi` (Linux) are declared deps now, so `setup_sherlock_env.sh`
> installs them automatically — no manual `pip install` needed.

Activate it in any later terminal with:
```bash
ml python/3.12.1 && source /home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate
```

### 2. Point heavy I/O at $SCRATCH, then re-fetch the CTX tiles
The CTX tiles (~12 GB) and outputs must NOT land in the 15 GB `$HOME`. Symlink the cache and
output dirs onto `$SCRATCH` (the relative paths in the code keep working, the bytes live on
Lustre):
```bash
mkdir -p $SCRATCH/hirise2ctx/{cache,map_region}
ln -sfn $SCRATCH/hirise2ctx/cache      $HOME/hirise2ctx/cache_v2
ln -sfn $SCRATCH/hirise2ctx/map_region $HOME/hirise2ctx/reports/map_region
```
Re-fetch the 7 block tiles from Murray Lab (~12 GB, a few minutes on Sherlock's fast network),
using config for the URL template + cache dir exactly as `scripts/run_stage2.py` does. The
`HIRISE2CTX_INSECURE_TLS=1` from step 1 must be set in this shell (Murray Lab sends an
incomplete cert chain — same reason as the checkpoint):
```bash
export HIRISE2CTX_INSECURE_TLS=1     # if not already set in this shell
ml python/3.12.1                     # provides libpython3.12.so.1.0 the venv links against
source /home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate
python - <<'PY'
from src.config import load_config
from src.ctx_retrieve import ensure_tile_cached
cfg = load_config("config_v2.yaml")
tmpl = cfg["ctx_mosaic"]["url_template"]
cache = cfg["cache_dir"]            # -> cache_v2 (symlinked to $SCRATCH)
for t in ["E0_N40","E4_N40","E4_N44","E8_N40","E8_N44","E12_N44","E16_N44"]:
    ensure_tile_cached(t, url_template=tmpl, cache_dir=cache)
    print("cached", t)
PY
```
*(Verify `load_config` returns those keys on your branch — `scripts/run_stage2.py` is the
canonical caller; mirror it if the accessor differs.)*

### 3. Parity gate — the de-risk, do not skip
This proves the GPU box reproduces the laptop's predictions, so a torch/CUDA/fp16 difference
can't silently corrupt the map. The reference was generated and uploaded from the laptop
(`models/deployable/parity_ref.npz`). On the **GPU node**:
```bash
python scripts/parity_check.py
# -> [PASS] predictions match ... safe to run scripts/map_region.py
```
`[PASS]` (exit 0) = go. A `[FAIL]`/nonzero exit means drift — stop and investigate before
spending GPU hours. (Tolerance is set for fp16; tiny differences are expected and pass.)

---

## Part C — Run the map

### C1. Throughput probe → size the job (optional but smart)
```bash
python scripts/map_region.py --tiles E4_N44 --limit-windows 4 \
    --out-dir $SCRATCH/hirise2ctx/map_region
```
Read the `~N tiles/s` lines. **Laptop baseline (RTX 5070, fp16): ~365 tiles/s → the full
7-tile block ≈ 15.4M tiles ≈ 11.7 GPU-h.** A Sherlock A100/H100 is ~2–3× faster (~4–6 h); an
older V100 is similar-to-slower than the laptop. Use your measured rate to decide `--time`.

### C2. Submit a run (batch, unattended, resumable)
```bash
sbatch run_region.sbatch --tiles E4_N44 E8_N44   # specific tiles, one GPU, sequential
sbatch run_region.sbatch --all                   # the full 26-tile map (skips done tiles)
```
`run_region.sbatch` requests 1 GPU / 8 CPUs / 32 GB / 12 h on the `gpu` partition and writes
GeoTIFFs to `$SCRATCH/hirise2ctx/map_region`. **It is checkpointed at the (tile, read-window)
level**: if the 12 h limit is hit or the job is pre-empted, just re-`sbatch` the same command —
it skips finished windows and tiles and continues (verified locally). Raise `--time` for fewer
resumes. **For the 19-tile expansion, prefer the parallel job array in C4** (much faster
wall-clock). `--all` is now the full **26-tile** regional map (was 7); it skips any tile whose
final GeoTIFF already exists, so it only computes what's missing.

### C3. Monitor
```bash
squeue -u $USER                       # is it queued (PD) or running (R)?
sacct -j <jobid>                      # exit status after it finishes
tail -f logs/h2c-map-<jobid>.out      # live progress: "[E4_N44] win 37/144 ... ~N tiles/s"
```

### C4. Expansion run — the 19 new tiles, PARALLEL (PLAN_RegionalMap §10 #5)
The regional map was widened to **26 tiles** (box lon[-10,10] lat[32,46] + the 2 kept NE tiles).
The first 7 already ran; the **19 net-new tiles** are independent, so fan them across a Slurm
**job array** (`run_region_array.sbatch`) instead of one sequential GPU — wall-clock ≈
sequential-GPU-h / concurrent-tasks (~13–19 GPU-h → ~2–3 h on 6 GPUs).

First **re-fetch the 19 new CTX tiles** (same snippet as step 2, expansion list):
```bash
export HIRISE2CTX_INSECURE_TLS=1
ml python/3.12.1                     # provides libpython3.12.so.1.0 the venv links against
source /home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate
python - <<'PY'
from src.config import load_config
from src.ctx_retrieve import ensure_tile_cached
EXPANSION_TILES = [  # = scripts/map_region.py EXPANSION_TILES (19 new tiles)
    "E-12_N32","E-12_N36","E-12_N40","E-12_N44","E-8_N32","E-8_N36","E-8_N40","E-8_N44",
    "E-4_N32","E-4_N36","E-4_N40","E-4_N44","E0_N32","E0_N36","E0_N44","E4_N32","E4_N36",
    "E8_N32","E8_N36"]
cfg = load_config("config_v2.yaml")
tmpl, cache = cfg["ctx_mosaic"]["url_template"], cfg["cache_dir"]
for t in EXPANSION_TILES:
    ensure_tile_cached(t, url_template=tmpl, cache_dir=cache); print("cached", t)
PY
```
Then submit the array (defaults to the 19 `EXPANSION_TILES`, 6-way, `--batch 256`):
```bash
mkdir -p logs
sbatch run_region_array.sbatch
# match the parity reference's batch exactly:   BATCH=96 sbatch run_region_array.sbatch
# fewer/more concurrent GPUs: edit  #SBATCH --array=0-5  (N tasks share the 19 tiles by stride)
```
Resumable like the sequential job: re-`sbatch` after a wall-clock/pre-emption — finished tiles
are skipped (final GeoTIFF exists), partial tiles resume mid-window. Monitor with
`squeue -u $USER` / `tail -f logs/h2c-map-arr-<arrayjob>_<task>.out`.

The array passes `--clean-partials`, so when it finishes **`$OUT` holds only the downloadable
products** — one flat folder of `<tile>_{prob,abundance,prob_raw}.tif` + `<tile>.json` for the 19
new tiles, no `partials/` clutter. Pull the whole folder back to the laptop in one shot:
```bash
# from the LAPTOP (the 7 already-run tiles are already in reports/map_region/).
# ~/hirise2ctx/reports/map_region is symlinked to $SCRATCH/hirise2ctx/map_region on Sherlock,
# so this home-relative path resolves there without needing $SCRATCH to expand:
rsync -av <sunet>@dtn.sherlock.stanford.edu:hirise2ctx/reports/map_region/ \
    ~/Documents/PhD/HiRiseToCTXBoulders/hirise2ctx/reports/map_region/
# (or scp the *.tif + *.json; or drag the folder via the OnDemand Files browser)
```

> **Batch + parity:** `--batch 256` better saturates the L40S and is ~parity-safe (the Fang ViT
> is per-sample). If you run the strict parity gate (C/B step 3), emit its reference at the same
> `--batch` — fp16 GEMM kernel choice can shift outputs by ~tol across batch sizes. `BATCH=96`
> keeps the existing reference exactly valid.

When all 26 tiles' GeoTIFFs are back on the laptop in `reports/map_region/`, run
`python scripts/map_mosaics.py` to stitch them (notebook 24 §2 **reads** those mosaics; it no
longer builds them — see the rewire note in `notebooks/_build_24.py`).
*(OnDemand also shows running jobs under **Jobs → Active Jobs**, and you can `tail` the log
file in the **Files** editor.)*

When done you'll have, per tile, `<tile>_prob.tif` (calibrated P(rich)),
`<tile>_abundance.tif` (fractional_area), `<tile>_prob_raw.tif` (QA) — single-band float32,
160 m/px, `NaN` = nodata.

### C5. Growing the map to a NEW region — plan-driven (the general path)

C4 hardcodes its 19 tiles in a bash array. Anything beyond the circum-Chryse block uses this
path instead, where the tile list is **data**, so extending the map never edits an sbatch file
or this document.

**Step 1 — plan it, on the laptop.** The box is snapped out to whole 4° Murray tiles, tiles
already rendered in any existing product are excluded, and every remaining tile's URL is
checked at the Murray Lab mosaic (it is not a complete planetary grid, and western tiles only
resolve under the zero-padded name `E-024_N28`, never the bare `E-24_N28`):

```bash
python scripts/plan_map_extent.py --lat 20 48 --lon -24 -4     --map-dirs reports/map_region reports/map_extended     --verify-urls --json reports/map_extended/plan.json
```

It prints the delivered footprint, the GPU-hours from `region_manifest.json`'s **own measured**
17–22 s/window, and the download budget. The current plan — the Xanthe/Chryse bridge,
lon[–24,–4] lat[20,48] — is **35 tiles: 8 adopted from `map_region`, 27 to render, ~19.8 GPU-h
(median) ≈ 3.3 h on 6 GPUs, 49.0 GB of CTX zips**.

**Step 2 — adopt the overlap, on the laptop.** The 8 tiles the new box shares with the shipped
map were rendered by the same head on the same lattice; re-rendering them would cost ~6 GPU-h
to reproduce bytes we already have, and (because fp16 GEMM kernel choice varies with `--batch`)
might not reproduce them exactly. Copy instead, verified both ends against each sidecar's own
`rasters[]` record:

```bash
python scripts/adopt_map_tiles.py --from reports/map_region --to reports/map_extended     --plan reports/map_extended/plan.json
python3 scripts/verify_map_download.py reports/map_extended --plan reports/map_extended/plan.json
```

**Step 3 — fetch the CTX zips, on a Sherlock login node** (internet; the GPU job does no
downloads). Resumable and per-tile fault-tolerant — a transient failure does not lose the rest:

```bash
export HIRISE2CTX_INSECURE_TLS=1        # Murray Lab serves an incomplete cert chain
ml python/3.12.1
source /home/groups/mlapotre/bamaro/envs/hirise2ctx/bin/activate
cd $HOME/hirise2ctx
python scripts/fetch_ctx_tiles.py --plan reports/map_extended/plan.json --dry-run   # budget
python scripts/fetch_ctx_tiles.py --plan reports/map_extended/plan.json
```

**Step 4 — submit.** `run_map_extended_array.sbatch` reads `to_render` out of the plan, **one
tile per array task**, so nothing in the sbatch changes when the box does:

```bash
mkdir -p logs
sbatch --array=0-26 run_map_extended_array.sbatch      # 27 tiles -> 0-26
# resubmit only what is left:
#   ONLY="E-24_N44 E-20_N44" sbatch --array=0-1 run_map_extended_array.sbatch
# a different plan:
#   PLAN=reports/map_other/plan.json sbatch --array=0-N run_map_extended_array.sbatch
```

It **inherits every hard-won setting from `run_rebuild_map_array.sbatch`**, which is the script
that actually produced the 8 tiles this product adopted. None of these are defaults:

* `--model-parent models/deployable_g2` + explicit `--calibration` + `--size-floor-basis`.
  `resolve_model_dir` picks `hits[-1]` **by name**, and the default parent `models/deployable`
  resolves to the **legacy** head `86c51a5dca220f63` — a completely different digest. Rendering
  with it would have made the 27 new tiles a different product from the 8 adopted ones,
  silently. Omitting `--size-floor-basis` also emits **no `SIZE_FLOOR_*` tags** (R84).
* A **preflight** that hashes the resolved head + calibration and refuses to start unless they
  match the digests recorded in the adopted tiles' sidecars (`29e833be…` / `290a8661…`). It
  costs seconds and is the only thing that catches this class of error before the GPU time.
* `--batch 96`, the parity reference's batch (256 buys nothing: 723 vs 730 img/s).
* **One tile per task.** In step 11 a task that overran took its remaining tiles down with it,
  losing 3 never-attempted tiles. One tile per task makes the blast radius one tile.
* `--constraint GPU_SKU:RTX_2080Ti` **and** `--require-device "2080 Ti"`.

⚠ **Pinning the GPU is the single biggest cost lever, and it is why `--time` can be 3 h.**
Per-window cost varies **11×** across the `gpu` partition: 2080 Ti **17.9 s/win = 0.72 h/tile**,
P100 91.7, TITAN Xp ~202 = **8.08 h/tile** — one TITAN Xp tile alone would exceed the wall, which
is exactly how the A1 array died. The Slurm constraint is the first defence; `--require-device`
is the second, because a feature name that is wrong or renamed fails *silently*. So: **~19.4
GPU-h total**, and wall-clock ≈ one tile (~45 min) plus queueing if the array runs wide (there
are 64 such cards). To accept a slower card, widen `REQUIRE_DEVICE` and raise `--time` together:

```bash
REQUIRE_DEVICE="P100" sbatch --array=0-26 --time=05:00:00 run_map_extended_array.sbatch
```

⚠ **`exit 0` means nothing on its own in this pipeline** — a failing tile no longer kills its
task. Read the `N/M tiles complete` tally in each log.

**Step 5 — bring it home and stitch.** Output lives at `$SCRATCH/hirise2ctx/map_extended`.
⚠ **Do not symlink `reports/map_extended` onto `$SCRATCH`** the way step 2 does for
`map_region`: that directory is tracked in git for `plan.json`, and the symlink would hide it
from the job. Verify on the cluster against the scratch path, then rsync into the repo dir on
the laptop:

```bash
# on Sherlock
python3 scripts/verify_map_download.py $SCRATCH/hirise2ctx/map_extended     --plan reports/map_extended/plan.json          # 8 "NO sidecar" until the adopted tiles arrive
# on the LAPTOP (plan.json and the 8 adopted tiles are already there)
rsync -avP <SUNetID>@dtn.sherlock.stanford.edu:'$SCRATCH/hirise2ctx/map_extended/*'     ~/Documents/PhD/HiRiseToCTXBoulders/hirise2ctx/reports/map_extended/
python scripts/verify_map_download.py reports/map_extended --plan reports/map_extended/plan.json
python scripts/map_mosaics.py --baseline reports/map_extended --layers abundance prob prob_raw
```

⚠ **`reports/map_region` and `reports/map_a1` stay frozen at 26 tiles.** Their footprint gate
(`n_finite == 26 × 1479² − 7,940`), their sidecar QA and their cell-for-cell arm parity are all
written against exactly those 26 tiles, and the A1 arm is not being extended (A1 was demoted to
a sensitivity arm on 2026-08-25). `map_extended` is a separate, growable product on the **same**
global R01 lattice, so the two can be compared or merged later by construction.

⚠ **Truth coverage thins fast outside circum-Chryse.** Of the 39-image cohort, 23 fall inside
the shipped 26-tile map but only **1** falls inside the new lon[–24,–4] lat[20,36] block
(`ESP_042964_2160`). South of ~32°N the map is extrapolation beyond where LOIO validation
exists — and it still carries the unmitigated source-frame artifact. Say so in any caption.

---

## Part D — Bring results home

- **OnDemand:** Files → `$SCRATCH/hirise2ctx/map_region` → select the `.tif`s → **Download**
  into the laptop's `reports/map_region/`.
- **SSH:**
  ```bash
  rsync -avP <SUNetID>@dtn.sherlock.stanford.edu:'$SCRATCH/hirise2ctx/map_region/*.tif' \
      reports/map_region/
  ```

Then on the laptop re-run `notebooks/24_regional_map.ipynb` — §2 auto-renders the abundance
mosaic from the returned GeoTIFFs — and proceed to the validation legs.

> ⚠️ **`$SCRATCH` is purged after 90 days of inactivity.** The GeoTIFFs are small (keep them
> on the laptop). The ~12 GB CTX cache and any embedding cache can be left to expire.

---

## Part E — F de-risk: the 10-frame ISIS timing test (PLAN_StripingArtifact)

Prices option **F** (per-source-frame inference) before committing: EDR download → `mroctx2isis`
→ `spiceinit web=yes` → `ctxcal` → `ctxevenodd` → `cam2map` on the 10 frames in
`reports/f_timing/frame_list.csv` (built + URL-verified on the laptop by
`scripts/f_edr_frame_list.py --verify`; ~2 GB total download). CPU-only — no GPU, and a
separate env from the map venv (ISIS ships only via conda channels — the USGS
`usgs-astrogeology` channel, deps from conda-forge; Sherlock discourages system conda, so
this uses **micromamba**, a single static binary).

```bash
# once: env + ~10 GB ISIS base data to $SCRATCH/isisdata. Use a COMPUTE session (sh_dev -c 4):
# login nodes kill micromamba's parallel install with EAGAIN (script warns if you try).
cd ~/hirise2ctx && git pull
bash setup_isis_env.sh

# once: targeted MRO kernel + calibration fetch (~1-2 GB; login node -- needs internet).
# Reads the kernel names the web service resolved into the first run's isis_steps.log.
bash f_fetch_kernels.sh

# then
sbatch run_f_timing.sbatch
# watch:   tail -f logs/h2c-f-timing-*.out
# result:  $SCRATCH/hirise2ctx/f_timing/timing.csv  (per-step timings + the 907-frame
#          regional / 86,571-frame global extrapolation printed at the end)
# home:    copy timing.csv back into reports/f_timing/ (OnDemand Files or rsync, as Part D)
```

Failure modes we hit on the way (all fixed in-repo, kept for recognition):
- **micromamba install dies with `Resource temporarily unavailable`** → you're on a login node;
  use `sh_dev -c 4` (per-user thread cgroup is the constraint, thread caps don't save you).
- **`spiceinit web=yes` → "The SPICE server returned incompatible SPICE data"** → the web
  service is version-pinned to a different ISIS release than the conda `isis` client; that's
  why the test runs `web=no` on local kernels from `f_fetch_kernels.sh`. For a NEW frame set,
  run once with `SPICE_WEB=yes sbatch run_f_timing.sbatch` to harvest the kernel names into
  the log, `bash f_fetch_kernels.sh`, then re-submit (default local).
- **`ctxcal` error naming a missing calibration file** → the targeted mro pull missed it;
  `f_fetch_kernels.sh` includes `calibration/**`, or pull the named file manually.
- Compute nodes DO have outbound internet (verified: `srun curl` → HTTP 206), so the EDR
  downloads inside the job are fine.

For the **F pilot** the same job doubles as the cube factory: `KEEP_CUBES=1 sbatch
run_f_timing.sbatch` keeps the projected `.map.cub`s (~36 GB scratch), then
`python scripts/f_pilot_extract_crop.py` (map venv) windows the 7 crop frames to small I/F
GeoTIFFs → tar → laptop `reports/f_timing/pilot_crops/` (analysis runs there on the local GPU).

## Part F — F pilot leg B: ISIS processing for the training cohort

Processes the ~40–80 CTX source frames that cover the 38-image training cohort so the
laptop can re-embed training windows from calibrated I/F frames and run the LOIO skill
gate (PLAN_StripingArtifact leg B).  CPU-only; same ISIS micromamba env as Part E.

**Step 0 (laptop) — build the frame list:**
```bash
conda run -n geospatial python scripts/f_leg_b_frame_list.py
# writes reports/f_leg_b/cohort_frame_list.csv   (unique EDR URLs)
#         reports/f_leg_b/obs_frame_map.csv       (obs_id -> PRODUCT_ID pairs)
#         reports/f_leg_b/cohort_obs_bounds.csv   (obs CTX-CRS window bounds)
# prints estimated Sherlock wall-clock
git add reports/f_leg_b/ && git commit -m "leg B frame list"
```

**Step 1 (Sherlock) — ISIS array job (~1 h wall on 24 tasks):**
```bash
cd ~/hirise2ctx && git pull
mkdir -p logs
sbatch run_f_leg_b.sbatch
# watch:   tail -f logs/h2c-f-legb-*.out
# result:  $SCRATCH/hirise2ctx/f_leg_b/*.map.cub  (one per frame, kept)
```
If the cohort frame count is much larger or smaller than ~50, adjust `--array=0-XX` in
`run_f_leg_b.sbatch` to keep wall-clock ≤ 1 h (target: 2-3 frames per task).

**Step 2 (Sherlock, MAP venv) — extract I/F crops:**
```bash
ml python/3.12.1
source /home/groups/mlapotre/$USER/envs/hirise2ctx/bin/activate
cd $HOME/hirise2ctx && git pull
python scripts/f_leg_b_extract.py            # default --resampling cubic
# writes $SCRATCH/hirise2ctx/f_leg_b/obs_crops_cubic/{obs_id}_{pid}_ifcrop.tif
# prints total extracted + any missing cubes
```
> 2026-07-05: the original run used bilinear (`obs_crops/`), which halved the crops'
> Nyquist power vs the mosaic (blur_check.csv, HF ratio 0.40) — cubic is now the
> default and writes to a resampling-specific dir so old crops can't be resume-skipped.

**Step 3 — transfer crops back to laptop:**
```bash
# on Sherlock:
tar cf obs_crops_cubic.tar -C $SCRATCH/hirise2ctx/f_leg_b obs_crops_cubic
# OnDemand Files: download obs_crops_cubic.tar  (~2 GB)
# on laptop:
mkdir -p reports/f_leg_b
cd reports/f_leg_b && tar xf ~/Downloads/obs_crops_cubic.tar
```

**Step 4 (laptop) — embed + LOIO gate:**
```bash
conda run --no-capture-output -n geospatial python -u scripts/f_leg_b_embed.py \
    --mapping minnaert --crops-dir obs_crops_cubic --store-suffix _c
# smoke test: add --smoke (2 images); mappings: perframe / global / minnaert
conda run -n geospatial python scripts/f_leg_b_loio.py --f-store fang_embeddings_f_minnaert_c
# prints PASS/FAIL gate + Δ median AUC vs baseline
```

Failure modes (in addition to Part E modes):
- **`_frames_{tile}.gpkg` missing for a tile** — `f_leg_b_frame_list.py` builds it from
  the cached seammap. If a tile has neither gpkg nor seammap, that obs_id is skipped
  and the script prints a warning; re-check `cache/ctx_tiles/`.
- **`no valid overlap in cube`** during extract — that frame doesn't actually cover the
  training image's CTX window (unusual; check the obs_frame_map.csv assignment).
- **Embed step "all-zero composite"** — no `*_ifcrop.tif` files for that obs_id in
  `reports/f_leg_b/obs_crops/`; confirm the tar transfer included that obs_id.

---

## Part G — F build sizing probe (PLAN_FBuild V1 + V5)

Before committing the 907-frame array, run 5 representative frames end-to-end to (V1) size the
array and (V5) decide per-frame vs per-row `cos^k(i)` (the audit 2026-07-23 within-frame incidence
ramp). Reuses the Part E ISIS timing kit; same ISIS micromamba env.

**Step 0 (laptop) — build the manifests + pick the 5 frames:**
```bash
conda run -n geospatial python scripts/f_build_framelist.py       # region_frame_list.csv (907) + frame_tile_map.csv (done 2026-07-23)
conda run -n geospatial python scripts/f_build_sizing_frames.py   # reports/f_build/sizing_frame_list.csv (5 frames, FPS over incidence/year/n_tiles)
git add reports/figures/region_frame_list.csv reports/figures/frame_tile_map.csv reports/f_build/ && git commit -m "fbuild manifests + sizing frames"
```

**Step 1 (Sherlock) — ISIS on the 5 frames, keeping the projected cubes (~15–20 min):**
```bash
cd ~/hirise2ctx && git pull && mkdir -p logs
sbatch run_f_build_probe.sbatch
# result: $SCRATCH/hirise2ctx/f_build_probe/timing.csv  +  {pid}.map.cub  (KEEP_CUBES=1)
```

**Step 2 (Sherlock) — cubes → GeoTIFF, transfer to the GPU box:**
```bash
cd $SCRATCH/hirise2ctx/f_build_probe
for c in *.map.cub; do gdal_translate -of GTiff "$c" "${c%.cub}.tif"; done   # ISIS3 driver
tar cf fbuild_probe.tar timing.csv *.map.tif
# OnDemand Files: download fbuild_probe.tar; on the laptop:
mkdir -p reports/f_build/probe_cubes && tar xf ~/Downloads/fbuild_probe.tar -C reports/f_build/probe_cubes
```

**Step 3 (GPU: laptop RTX 5070 or a Sherlock L40S) — measure V1 + V5:**
```bash
conda run --no-capture-output -n geospatial python -u scripts/f_build_sizing_probe.py \
    --frames-dir reports/f_build/probe_cubes \
    --timing-csv reports/f_build/probe_cubes/timing.csv
# writes reports/figures/fbuild_sizing_probe.csv + prints:
#   V1 -> tiles/frame, s/frame, 907-frame tile count + GPU-h + ISIS CPU-h + peak scratch
#   V5 -> per-frame within-frame ramp % (measured vs geometry-predicted) + the verdict:
#         <~0.5% keep per-frame cos^k(i); >=~1% switch Stage B to per-row cos^k(i(lat)) FIRST
```
(RTX 5070 → L40S s/frame scaling ≈ 1.0; refine `--gpu-scale` from a parity window if needed.)

**Then:** fold the V5 verdict into PLAN_FBuild §3 (scalar vs per-row) and the V1 numbers into the
array `--array=` sizing, and proceed to Stage A on all 907 (`run_f_leg_b.sbatch` pattern, scaled).

---

## Part H — F build Stage A: ISIS on all 907 region frames (PLAN_FBuild §2)

Calibrate + project every region frame to `{PRODUCT_ID}.map.cub`. Reuses the leg-B array worker
(`f_leg_b_process.sh`, now `FRAME_LIST`-parameterized + summed-frame-safe) via
`run_f_region_stagea.sbatch`. CPU-only; resumable (skips frames whose cube exists). Probe-confirmed
cost ~200–330 CPU-h ⇒ ~7–9 h wall on 32 tasks. Scratch is ample (100 TB), so all 907 cubes are kept.

**Step 0 (laptop) — the frame list is already built + committed:**
```bash
# reports/figures/region_frame_list.csv (907) came from scripts/f_build_framelist.py (2026-07-23).
# nothing to do unless it changed; just make sure Sherlock has it via git pull below.
```

**Step 1 (Sherlock) — first pass:**
```bash
cd ~/hirise2ctx && git pull && mkdir -p logs
sbatch run_f_region_stagea.sbatch
# watch:  squeue --me ;  tail -f logs/h2c-region-A-*_*.out
```

**Step 2 — census the results** (per-task status CSVs):
```bash
cd $SCRATCH/hirise2ctx/f_region
cat status_*.csv | awk -F, 'NR>1 && $1!="product_id"{n++; c[$11]++} END{print n" frames:"; for(k in c) print "  "k": "c[k]}'
ls *.map.cub | wc -l    # cubes produced so far
```

**Step 3 — fill the SPICE kernel gap and resume** (the July mirror lacks some 2018+ CKs; the
first pass logs exactly which by name):
```bash
cat $SCRATCH/hirise2ctx/f_region/isis_*.log > /tmp/region_isis.log
cd ~/hirise2ctx && bash f_fetch_kernels.sh /tmp/region_isis.log   # fetches the named CK/SPK
sbatch run_f_region_stagea.sbatch                                 # resumes; fills spiceinit_fail holes
```
Repeat Steps 2–3 until the census is clean or only a genuinely-unrecoverable handful remain (a few
failures out of 907 is expected — leg B saw 1/81; the graph's median degree 7 means an isolated
missing frame costs coverage, not the gauge). **Record the final failure list** — those tiles get
patched with the mosaic + flagged in the H6 provenance layer (§V4), not blocked on.

**Failure modes seen in the probe (all handled):**
- `spiceinit_fail` "Spice file does not exist [...ck/mro_sc_psp_YYMMDD...]" — missing kernel → Step 3.
- `evenodd_fail` — should no longer occur: the worker now skips `ctxevenodd` when `SpatialSumming>1`
  (you'll see `SpatialSumming=N -> skip ctxevenodd` in the log) and projects the calibrated cube.
- `download_fail` (instant, 0-byte) — transient PDS hiccup (NOT quota; scratch is 0.5/100 TB); the
  `--retry`/`--retry-delay` usually rides it out, and a resume re-tries the frame.

**When Stage A is clean → Stage B** (embed + infer, per-row `cos^k(i(lat))` mapping): that's the next
build stage (needs V2 incidence-for-907 + V3 parity first). Cubes stay on scratch as its input.

---

## Part I — F build Stage B: embed + infer, then hand off to Stage C (PLAN_FBuild §3–4)

Stage B reads the **GeoTIFFs** (`{PRODUCT_ID}.map.tif` from `run_f_region_tif.sbatch` — GDAL's ISIS3
driver segfaults on large windowed `.cub` reads, DECISIONS 2026-07-27) and writes one
`{PRODUCT_ID}.npz` = `{TI, TJ, prob}` per frame on the global 160 m tile grid, plus a `.json`.
6-GPU array, map venv, ~33 L40S-h total, **resumable** (skips frames whose npz exists).

```bash
cd ~/hirise2ctx && git pull
sbatch run_f_region_tif.sbatch          # once: .map.cub -> .map.tif
sbatch run_f_region_stageb.sbatch       # re-submit as needed; 8 h wall is short of 151 frames/task
ls $SCRATCH/hirise2ctx/f_region_logits/*.npz | wc -l     # progress out of 907
```

**Hand-off to Stage C** — Stage C is a laptop step (~10 min, no GPU), so bring the logits home. They
are small: ~2–3 MB/frame, **~2 GB for all 907**.

```bash
cd $SCRATCH/hirise2ctx && tar cf f_region_logits.tar f_region_logits
# OnDemand Files: download f_region_logits.tar; on the laptop:
tar xf ~/Downloads/f_region_logits.tar -C reports/     # -> reports/f_region_logits/
& $conda run --no-capture-output -n geospatial python -u scripts/f_region_stagec.py
```

Stage C can also run on a **login node** (`python scripts/f_region_stagec.py --logits-dir
$SCRATCH/hirise2ctx/f_region_logits`) — pure numpy/scipy — but the geology half of the trend guard
needs the cached MOLA/THEMIS regional rasters, which live on the laptop. Run it at home unless you
only want the solve. It is safe to run against a **partial** Stage B: it censuses the missing frames
to `fbuild_stagec_missing_frames.csv` and says so, so an early read costs nothing.

---

## Part J — bring the logits home and run Stages C + D (PLAN_FBuild §4-§5.1)

**Stage B is COMPLETE: 906/907 npzs** (the one permanent failure is an H6 hole to patch with the
mosaic). Everything downstream is a laptop step — pure numpy/scipy/rasterio, no GPU — so the whole
remaining build is a transfer plus ~30 minutes of local compute.

**1. Transfer (~2 GB).** On Sherlock:
```bash
cd $SCRATCH/hirise2ctx && tar czf ~/f_region_logits.tgz f_region_logits
ls -l ~/f_region_logits.tgz     # OnDemand Files -> download
```
On the laptop:
```powershell
tar xzf ~/Downloads/f_region_logits.tgz -C reports/     # -> reports/f_region_logits/ (906 npz + json)
```

**2. Stage C — the H4 solve** (~10 min; see §4). Partial-safe, so it can be run before the last
frames land:
```powershell
& $conda run --no-capture-output -n geospatial python -u scripts/f_region_stagec.py
#   -> reports/figures/fbuild_stagec_offsets.csv   the per-frame logit offsets
#      reports/figures/fbuild_trend_guard.{csv,png}  the verdict: full vs residual-only
```
**Read the verdict before Stage D.** If it is `AMBIGUOUS` (`apply=full_pending_ruling`), Stage D
deliberately writes no headline map — that is §0.1 guard 1, and the call is Brian's (§7 Q3).

**3. Stage D — composite** (~10 min). Needs the F-path calibrator (already banked) and writes on the
mosaic map's exact grid into a *separate* directory, so `reports/map_region/` survives as the §5.1
comparison object:
```powershell
& $conda run --no-capture-output -n geospatial python -u scripts/bank_calibration_f.py   # once; done
& $conda run --no-capture-output -n geospatial python -u scripts/f_region_staged.py
#   -> reports/map_fbuild/{tile}_{h1only,full,resid}_{prob_raw,prob,abundance,overlap_dp}.tif
#      + {tile}_{n_frames,primary_frame,incidence,offset_source}.tif   (H6 provenance)
#      + {tile}_{variant}_prob_partition.tif                          (gate 1's scoring layer)
```

**4. The gates** (~10 min for 26 tiles at `--null-draws 12`):
```powershell
& $conda run --no-capture-output -n geospatial python -u scripts/f_region_gates.py
#   -> fbuild_gate{1..6}*.csv + fbuild_gates.json + fbuild_gate4_choropleth.png
```
Gate 1's mosaic baseline is already banked (median-window partition η² **0.1222** against its own
rotation-null p95 **0.0676**, ratio 1.65 — the artifact). The F rows are what this run adds.

**5. §5.1 comparison.** The A1 row needs two GPU steps first — there is **no A1 raster on disk at any
extent**, and A1 renormalises raw CTX DN, so there is no post-hoc path:
```powershell
# ~5-7 GPU-h on the 9 CTX-equipped tiles (the only footprint A1 can cover without ~30 GB of downloads)
& $conda run --no-capture-output -n geospatial python -u scripts/striping_a1_map.py
# ~1 h GPU: A1 skill on the SAME 36 images the F rows use (post-hoc row filtering cannot fix the
# different training regime)
& $conda run --no-capture-output -n geospatial python -u scripts/striping_a1_loio.py `
    --restrict-store fang_embeddings_f_minnaert_center --tag _36
& $conda run --no-capture-output -n geospatial python -u scripts/f_map_compare.py
#   -> reports/figures/fbuild_vs_mosaic_vs_a1.{csv,png} + fbuild_cost_ledger.csv
```
`f_map_compare.py` runs without the A1 map and simply leaves that row blank, so the mosaic-vs-F
read is available immediately; the A1 column is what prices the build against the fallback.

---

## Going global later

`scripts/map_region.py` is tile-list-driven, so global inference = feed the full Murray tile
index instead of the 7-tile `--all` list (and re-fetch those tiles in step 2). Same driver,
same checkpointing. Size `--time` to your measured rate, or split into several jobs by
tile-group and submit them in parallel (each writes independent per-tile GeoTIFFs).

## Troubleshooting (incl. everything we hit on the first port — all fixed in-repo now)

- **`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`** (on the tile /
  checkpoint download) — Murray Lab & Zenodo send an **incomplete cert chain** and Linux
  OpenSSL won't auto-fetch the missing intermediate (Windows does, so it works on the laptop).
  Fix: `export HIRISE2CTX_INSECURE_TLS=1` before steps 1–2 (skips verification for these
  public, fixed-URL downloads only; off by default). Not a proxy — the issuer is a real CA
  (InCommon).
- **`python: error while loading shared libraries: libpython3.12.so.1.0`** — you activated the
  venv without loading the python module, so the interpreter's shared lib isn't on the library
  path. Fix: `ml python/3.12.1` **before** `source .../activate` (every fresh shell; the sbatch
  scripts already do this).
- **`No module named venv`** during setup — the python module exposed `python3` but not bare
  `python`; the setup script now builds the venv with the verified `python3`. (Python 3.12.1
  is correct — don't switch.)
- **`No module named 'truststore'` / `ModuleNotFoundError`** — `truststore`+`certifi` are now
  declared deps, installed by `pip install -e .`. If on an old checkout: `pip install truststore certifi`.
- **`No module named 'yaml'` / `'attr'` / `'certifi'` (or rasterio fails importing `attr`)** —
  venv drift: the binary geo stack installed but the dependency closure (`pip install -e .`,
  setup step 59) never finished, so pure-Python deps are missing. Fix: **`pip install -e .`** —
  it pulls the missing wheels (`pyyaml`, `certifi`, `attrs`, rasterio's `affine`/`click`/`cligj`/
  `snuggs`…) without rebuilding the already-installed binary geo packages. Verify with
  `python -c "import yaml, certifi, attr, rasterio, numpy, torch; print('env OK')"`.
- **`No module named 'typing_extensions'` / `'jinja2'` at `import torch`** — torch's own
  pure-Python deps are missing (the same drift; `pip install -e .` doesn't cover them since torch
  is installed separately). Run **`pip check`** to list every gap at once, then
  `pip install typing_extensions jinja2` (and any other torch line `pip check` reports —
  `sympy networkx filelock fsspec`). The many `jupyter*`/`pytest` lines from `pip check` are
  **harmless for inference** — notebooks/tests run on the laptop, not Sherlock. Confirm with
  `python -c "import torch; print(torch.__version__)"`.
- **scipy/scikit-learn/rasterio building from source / `OpenBLAS not found` / `gdal-config`** —
  Sherlock's old glibc can't load the newest manylinux_2_28 wheels and has no system
  GDAL/OpenBLAS. The setup script's `--only-binary` picks older compatible wheels; if you see
  this, you're not using the current `setup_sherlock_env.sh` (`git pull`).
- **`os has no attribute add_dll_directory`** — a Windows-only DLL shim leaked to Linux; fixed
  (guarded behind `os.name == "nt"`). `git pull` if you see it.
- **`cuda_available False`** — your terminal isn't on a GPU node. Relaunch the OnDemand app
  (or `sh_dev`) with Partition = `gpu`, GPUs = 1.
- **Job stuck in `PD` (pending)** — the `gpu` partition is busy; wait, or check `squeue`. GPU
  queues can be slow at peak times.
- **`tile zip missing`** — step 2 didn't fetch that tile; re-run the fetch snippet (with the
  TLS flag set).
- **Out of space on `$HOME`** — you skipped the `$SCRATCH` symlinks in step 2; the 12 GB
  cache landed in your 15 GB home. Remove `cache_v2`, make the symlink, re-fetch.
- **geopandas/pyogrio I/O error** — `pip install pyogrio` into the venv (Linux wheel).

## What we deliberately ignore from `sherlock_hirise2ctx_runbook.md`
That older doc ports the **whole CPU pipeline** (Stages 1–5: detections → CTX sweep → coreg →
labels → splits) with **CPU-only** torch. We don't rebuild the dataset and we **want CUDA**
torch for the embedder, so only its Sherlock basics (storage, Slurm, login-vs-compute) carry
over; its stage commands and CPU env do not.
