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

When all 26 tiles' GeoTIFFs are back on the laptop in `reports/map_region/`, notebook 24 §2
stitches them automatically (it globs `*_abundance.tif`).
*(OnDemand also shows running jobs under **Jobs → Active Jobs**, and you can `tail` the log
file in the **Files** editor.)*

When done you'll have, per tile, `<tile>_prob.tif` (calibrated P(rich)),
`<tile>_abundance.tif` (fractional_area), `<tile>_prob_raw.tif` (QA) — single-band float32,
160 m/px, `NaN` = nodata.

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
# once (login or sh_dev node; downloads ISIS + ~10 GB base data to $SCRATCH/isisdata)
cd ~/hirise2ctx && git pull
bash setup_isis_env.sh

# then
sbatch run_f_timing.sbatch
# watch:   tail -f logs/h2c-f-timing-*.out
# result:  $SCRATCH/hirise2ctx/f_timing/timing.csv  (per-step timings + the 907-frame
#          regional / 86,571-frame global extrapolation printed at the end)
# home:    copy timing.csv back into reports/f_timing/ (OnDemand Files or rsync, as Part D)
```

First-run failure modes to expect: `spiceinit web=yes` needs outbound HTTPS from compute nodes
(the map's `/vsicurl/` reads already work there, so it should too); a `ctxcal` error naming a
missing calibration file means the targeted `downloadIsisData mro -- --include "calibration/**"`
pull didn't cover it — fetch the named file or the full `mro` area and re-submit (the driver
skips nothing: each frame is independent, failures are recorded per-row in timing.csv).

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
