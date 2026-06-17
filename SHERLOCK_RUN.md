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

### A1. Get a terminal on a GPU compute node
This single session gives you a shell *and* a file browser for all of setup + the parity
check + the throughput probe.

- **Interactive Apps → JupyterLab.** In the form set: **Partition = `gpu`**, **GPUs = 1**,
  CPUs = 8, Memory = 32 GB, Time = 3 hours. Click **Launch**, wait for the queue, then
  **Connect to JupyterLab**.
- In JupyterLab open **File → New → Terminal**. That terminal is running **on a GPU node** —
  exactly where the setup and parity check should run.

  *(Alternative for a pure shell with no Jupyter: Dashboard → **Clusters → Sherlock Shell
  Access** gives you a login-node shell. Good for `sbatch`/`git`, but it is NOT a GPU node,
  so don't run the parity check there — use the JupyterLab terminal for anything that needs
  the GPU.)*

### A2. Upload the trained artifacts via the Files browser
These three live only on your laptop and are gitignored (so `git clone` won't bring them).
On the **laptop**, bundle them into one file first (Git Bash):
```bash
cd /c/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise2ctx
tar -czf h2c_artifacts.tgz \
    models/deployable/86c51a5dca220f63 \
    models/deployable/calibration.npz \
    models/deployable/parity_ref.npz
```
In OnDemand: **Files → Home Directory**, navigate into `hirise2ctx/` (after step B0/A3 clones
it), click **Upload**, drop `h2c_artifacts.tgz`. Then in the terminal:
```bash
cd $HOME/hirise2ctx && tar -xzf h2c_artifacts.tgz && rm h2c_artifacts.tgz
ls models/deployable/      # should show 86c51a5dca220f63/  calibration.npz  parity_ref.npz
```

### A3. Now do the shared setup
Continue in the JupyterLab terminal with **Setup steps 0–3** below.

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
bash setup_sherlock_env.sh
```
What it does, and what to expect: loads `python/3.12.1`, creates a venv at
`/home/groups/mlapotre/bamaro/envs/hirise2ctx` (backed up), installs **CUDA** PyTorch +
the project deps, and downloads the 341 MB Fang-ViT checkpoint from Zenodo into
`models/pretrained/`. It ends by printing `cuda_available True` and your GPU's name — if you
see `False`, you launched the terminal on a non-GPU node (relaunch the app with Partition =
`gpu`, GPUs = 1). First run takes a few minutes (it compiles nothing, just downloads wheels).

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
Re-fetch the 7 block tiles from Murray Lab (a few minutes on Sherlock's fast network),
using config for the URL template + cache dir exactly as `scripts/run_stage2.py` does:
```bash
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

### C2. Submit the full block (batch, unattended, resumable)
```bash
sbatch run_region.sbatch --all
```
This requests 1 GPU / 8 CPUs / 32 GB / 12 h on the `gpu` partition and writes GeoTIFFs to
`$SCRATCH/hirise2ctx/map_region`. **It is checkpointed at the (tile, read-window) level**
(1008 windows for the block): if the 12 h limit is hit or the job is pre-empted, just
`sbatch run_region.sbatch --all` **again** — it skips finished windows and tiles and
continues (verified locally). Raise `--time` in the script for fewer resumes.

### C3. Monitor
```bash
squeue -u $USER                       # is it queued (PD) or running (R)?
sacct -j <jobid>                      # exit status after it finishes
tail -f logs/h2c-map-<jobid>.out      # live progress: "[E4_N44] win 37/144 ... ~N tiles/s"
```
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

## Going global later

`scripts/map_region.py` is tile-list-driven, so global inference = feed the full Murray tile
index instead of the 7-tile `--all` list (and re-fetch those tiles in step 2). Same driver,
same checkpointing. Size `--time` to your measured rate, or split into several jobs by
tile-group and submit them in parallel (each writes independent per-tile GeoTIFFs).

## Troubleshooting

- **`cuda_available False`** — your terminal isn't on a GPU node. Relaunch the OnDemand app
  (or `sh_dev`) with Partition = `gpu`, GPUs = 1.
- **Job stuck in `PD` (pending)** — the `gpu` partition is busy; wait, or check `squeue`. GPU
  queues can be slow at peak times.
- **`tile zip missing`** — step 2 didn't fetch that tile; re-run the fetch snippet.
- **Out of space on `$HOME`** — you skipped the `$SCRATCH` symlinks in step 2; the 12 GB
  cache landed in your 15 GB home. Remove `cache_v2`, make the symlink, re-fetch.
- **geopandas/pyogrio I/O error** — `pip install pyogrio` into the venv (Linux wheel).

## What we deliberately ignore from `sherlock_hirise2ctx_runbook.md`
That older doc ports the **whole CPU pipeline** (Stages 1–5: detections → CTX sweep → coreg →
labels → splits) with **CPU-only** torch. We don't rebuild the dataset and we **want CUDA**
torch for the embedder, so only its Sherlock basics (storage, Slurm, login-vs-compute) carry
over; its stage commands and CPU env do not.
