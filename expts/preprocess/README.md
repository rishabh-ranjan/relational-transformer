# Preprocessing a collection

Rebuilds a `*-preprocessed` dataset from its raw one and publishes it back as a
replacement. Two collections, one pipeline:

| collection | databases | raw | preprocessed |
|---|---|---|---|
| `the-join` | 639 | 28 GiB | ~1.4 TiB |
| `relbench` | 7 | 10 GiB | ~230 GiB (plus a `legacy/` tree) |

Four commands, in order. Which collection they act on is the block of constants
at the top of `submit.py` — edit it; the other sits below it commented out.

```bash
# 0. make room under /dfs/user/$USER: ~1.5 TiB for the-join, ~250 GiB for relbench
df -h /dfs/user/$USER

# 1. fetch the raw collection once
pixi run python expts/preprocess/download.py

# 2. submit. Re-run any time -- it submits only what is left, and only
#    databases whose raw files have all arrived, so it is safe to start while
#    step 1 is still running and to re-run as more land.
pixi run python pixi run python expts/preprocess/submit.py

# 3. watch it
pixi run python expts/preprocess/status.py
watch -n60 pixi run python expts/preprocess/status.py

# 4. verify, write the task lists, and publish
pixi run python expts/preprocess/finalize.py upload
```

Everything a collection differs by — its repos, a task list that cannot be
recomputed, directories the upload must not delete — is that one block. Adding a
third is a third block.

Step 2 refuses to run from a dirty or unpushed tree, and says so before it
submits anything — jobs clone the commit you submit from, so commit first.

To run the whole thing to completion unattended, loop step 2 (it is idempotent
and cheap) until `status.py` reports 639, then do step 4.

## What each piece is

| file | what it does |
|---|---|
| `download.py` | one resumable git-lfs fetch of the raw collection; `--repair` mends it |
| `preprocess.py` | the job targets: `rustler` (cpu-only), `embed` (GPU), `legacy` |
| `submit.py` | sizes each job, submits the stages, holds the resource numbers |
| `status.py` | progress and ETA, measured in bytes |
| `finalize.py` | `verify`, `task-lists`, `upload` |
| `sizes.py` / `sizes-<collection>.json` | output **and text** bytes per database, from the previous build |
| `rt-j-dbs.json` | the 475 databases rt-j trains on, carried forward |

`submit.py` holds the collection and the two values every stage must agree on;
every resource number is written at the call that passes it.

### The `legacy/` tree

`relbench` also publishes `legacy/`: the same databases under RT-v1's boolean
typing, which the released RT-v1 checkpoints need. `submit.py` builds it
alongside the main sweep, into its own directory, and `finalize.py upload`
verifies it together with the build — **a problem in either publishes
neither**. The Hub keeps the previous version, whole, until there is a whole new
one to replace it with.

## Two stages, and why

A database is preprocessed by **two jobs**, not one:

* **`rustler`** — one cpu, **no GPU**, memory from the expected output. Measured
  over 200 databases: `TotalCPU` equals `Elapsed` on every one, so the stage is
  single-threaded, and `MaxRSS` is about twice a database's output. It is ~70%
  of the work and near-100% on the big ones.
* **`embed`** — **six GPUs in one job**, 24 cpus, memory per GPU.
  sentence-transformers runs a worker per device, so one rank wants all of them
  (`ntasks=1`), not a rank each.

Together in one job they get neither. The first sweep asked for 8–48 cpus and
80–500 G per database and used **one cpu and 1.5–62 G**: 562 cpus allocated
across five nodes against a total `CPULoad` of 13, and concurrency capped at 22
by our own requests. Worse, every rustler stage held a GPU it never touched, and
50 GPUs across five nodes cap the sweep at 50 databases however much memory is
free.

Split, the cpu stage runs as wide as the nodes have cpus, and the GPU stage is a
short queue behind it — submitted in the same pass with `--dependency=afterok`,
so nothing polls and there is no second command to remember.

One job per database rather than a job array over shards, because the work is
lopsided: the median database preprocesses to ~43 MiB, the largest
(`join-tpch`) to 76 GiB, and 20 of the 639 are half the bytes. Round-robin
sharding — what `rt.preprocess.many` does — splits that about as badly as it
can be split. This is only affordable because roach shares one clone per commit
per node: the first job on a node pays ~5 minutes, every job after it starts in
**1–2 seconds**, so 1278 jobs cost about a minute of makespan in total.

## Resuming, failures, preemption

**Re-run `submit.py`.** It is the resume, for both stages at once: it works out
from what is on disk what each database needs next, so a failed job, an
out-of-memory kill, a node going down, a preemption or an interrupted sweep all
have the same fix.

* rustler is done when `text.json` exists (it is written last).
* the database is done when `meta.json` names an embedding file that exists.
  Not when its directory exists — a job preempted between the stages leaves a
  directory that looks finished.

`finalize.py verify` checks all eight expected files for all 639 and refuses to
upload if any is missing or empty; `upload` runs `verify` itself.

Jobs run under `il-lo`, which is preemptible. Slurm requeues them and both
targets are idempotent, so preemption costs the work in flight and nothing else.

Logs are `/dfs/user/ranjanr/slurm-logs/preprocess/<run_id>_<jobid>.out`.

## What the code cannot handle

Everything else this build ran into is fixed in the code and cannot recur
through the sanctioned path: the Hub's rate limits (`download.py` fetches by
git-lfs, the embedder is prefetched once per node), both kinds of
out-of-memory in the embedding stage (chunked encoding, memory sized from the
database), half-downloaded raw data (`submit.py` checks every file against the
Hub's recorded size), a stale `SLURM_CPUS_PER_TASK` (roach unsets it), and an
embedding that would take two days on one GPU (it takes six). The reasoning
lives next to each, in the module that implements it.

What is left is policy, which no amount of code will fix:

* **`il-lo` is preemptible.** Nothing to do about it: both stages are
  idempotent, so a requeued job costs the work in flight and nothing else.

Two design constraints hold the rest of it up, and a future change could
quietly break either:

* **Nothing writes into `RAW_DIR` except `download.py`.** The size check catches
  corruption after the fact; not creating it is better. Use
  `download.py --repair` to mend, never a copy from somewhere else.
* **Do not run downloaders in parallel.** The Hub's limit is not per-process,
  and extra processes turn a slow fetch into one that fails outright.

## The cluster, as measured

| node | cpus | memory | GPUs |
|---|---|---|---|
| hyperturing1 | 252 | 2011 G | 10 × RTX8000 48G |
| hyperturing2 | 252 | 2011 G | 10 × RTX8000 48G |
| turing1 | 80 | 754 G | 10 × 2080Ti 12G |
| turing2 | 80 | 754 G | 10 × 2080Ti |
| turing3 | 80 | 1448 G | 10 × 2080Ti |

`/dfs` measured ~700 MB/s buffered write and ~680 MB/s read from one node, so
writing the output there is not the bottleneck. `MaxMemPerCPU` is 10700M but is
not enforced against a per-node `--mem` (checked with `sbatch --test-only`: a
1-cpu job may ask for 200G), which is why the rustler stage can take one cpu
regardless of how much memory it needs.

**`DefMemPerGPU=240000M` limits how many GPUs a job can have, and `--mem` does
not lift it.** The partition applies that default when deciding whether a job
fits, so the most GPUs a job can hold is `RealMemory / 240000M` — **3** on a
754G turing, **6** on turing3, **8** on a hyperturing — however little memory it
actually asks for. It looks like contention (`Requested node configuration is
not available` on a node with nothing running on it) and is not. Use
`--mem-per-gpu`, which replaces the default: with it, an idle 754G node accepts
a request for 8 GPUs.

### Embedding throughput, measured

2M texts, same data on both node types:

| | 1 GPU | 6 GPUs | speedup | efficiency |
|---|---|---|---|---|
| hyperturing2 — Quadro RTX 8000 | 929 texts/s | 3,769 texts/s | 4.06× | 68% |
| turing3 — GeForce RTX 2080 Ti | 849 texts/s | 3,643 texts/s | 4.29× | 71% |

Two things follow. **Six GPUs are worth about four**, which is why the stage
takes a node's worth rather than one card — the ten databases that are 88% of
the stage each finish in a quarter of the time instead of queueing behind a
single GPU. And **the card barely matters**: 9% between an RTX 8000 and a 2080
Ti, so there is nothing to gain by routing text-heavy databases at the better
hardware.

The 68–71% efficiency is per-chunk fan-out overhead. Larger `CHUNK` in
`rt.preprocess.embed` would recover some of it, at the cost of peak memory.

## Progress and ETA

`status.py` measures **bytes, not databases** — counting databases would report
the sweep as nearly done with a quarter of the work left. Each database is
weighted by what the previous build's output measured. It withholds an ETA until
five databases have finished inside the rate window, because extrapolating from
two on work spanning three orders of magnitude produces answers off by weeks.

The `stuck` line is databases that are neither finished nor being retried — not
every job that ever failed, which would keep reporting a database that failed
once and succeeded on resubmit.

## What the stages actually cost

Measured over this build, from the `= <db>: ...` line each job prints. Numbers
are cpu/GPU seconds of real work, not wall clock — with hundreds of jobs at once
the sweep finishes far faster than these totals suggest.

| | rustler | embed |
|---|---|---|
| total | 3.8 h | 10.6 h |
| share of the work | **26%** | **74%** |
| median database | 2 s | 4 s |
| mean | 47 s | 128 s |
| slowest | 1743 s | 7173 s |

**Both stages are dominated by a handful of databases, but not the same ones.**

| | rustler | embed |
|---|---|---|
| top 1 database | 12.9% | 18.9% |
| top 5 | 46.5% | **68.5%** |
| top 10 | 64.5% | **87.9%** |
| top 20 | 79.7% | 93.1% |
| top 50 | 95.2% | 96.4% |

The embed stage is the more skewed and the more expensive: ten databases are 88%
of it. And the two rank differently, because they scale on different things:

* **rustler tracks output size.** Slowest are `join-se-electronics` (81 GiB out,
  1743 s), `join-tpch` (72 GiB, 1332 s), `join-se-english` (57 GiB, 1225 s).
* **embed tracks text volume**, which output size predicts poorly. Slowest are
  `join-open-food-facts` (7173 s from only 10 GiB of output),
  `join-bird-codebase-comments` (5983 s, 14 GiB), `join-unpaywall` (4727 s,
  13 GiB). `join-overture-maps` has 8 GiB of `text.json` and took over four
  hours.

Two consequences worth keeping in mind:

1. **`sizes.json` is a good predictor for rustler and a poor one for embed.** It
   is what sizes both stages' resources, which is why the embed tiers are
   generous rather than tight — a database with modest output and enormous text
   is exactly the case that gets under-provisioned, and every embed failure in
   this build (two OOMs and a timeout) was one of those. If this is ever tuned
   properly, size the embed stage on `text.json` bytes, which are known once
   rustler has run.
2. **The tail is free.** The median database costs 2 s + 4 s. Roughly 590 of the
   639 together are a few percent of the work, so effort spent scheduling them
   more cleverly buys nothing; all of the makespan is in the top ~20.

For a whole-collection estimate: ~14.5 hours of single-stream work, which at the
concurrency these five nodes allow (cpus for rustler, ~40 GPUs for embed) lands
in well under an hour of wall clock once the raw data is present. The raw
download is the longer pole and is not parallelisable — see `download.py`.

## Publishing

`finalize.py upload` verifies, regenerates `db-task-lists/`, pushes with
`upload_large_folder`, then deletes the database directories the Hub has and
this build does not. That last step is what makes it a replacement: the previous
build has 650 databases, 11 of which were dropped from the raw collection for
overlapping RelBench and DBInfer sources. Root files (`README.md`,
`.gitattributes`) are not touched.

Expect a full transfer rather than a cheap one: rustler's output changed since
the previous build (`nodes.rkyv` came out ~10% larger on the database compared),
so content hashing will not skip much.

`all.json`, `forecast.json` and `autocomplete.json` are derived from the metas.
`rt-j.json` is **not** derivable — it is a curated whitelist of 475 databases
(126 excluded wholesale, none partially) — so the database list is kept here in
`rt-j-dbs.json` and the pairs are rebuilt against it. `task-lists` prints any
curated database missing from the build, which is how you notice the whitelist
has drifted.

## Regenerating `sizes.json`

```bash
pixi run python expts/preprocess/sizes.py
```

Reads file sizes from the published repo, downloading nothing. It is a
prediction, used for job sizing and progress weighting; a database not in it is
submitted at the largest tier, on the grounds that unknown is not the same as
small.
