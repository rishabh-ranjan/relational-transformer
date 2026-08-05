# Preprocessing the Join

Rebuilds `stanford-star/the-join-preprocessed` from `stanford-star/the-join`:
639 databases, ~1.34 TiB of rustler artifacts and text embeddings, one slurm job
per database across five nodes, then published back to the Hub as a replacement.

Four commands, in order. Nothing else needs deciding.

```bash
# 0. make room: the build needs ~1.4 TiB free under /dfs/user/$USER
df -h /dfs/user/$USER

# 1. fetch the raw collection once (~28 GiB, ~30 min)
pixi run python expts/preprocess/download.py

# 2. submit one job per database (re-run any time; it submits only what is left)
pixi run python expts/preprocess/submit.py --dry-run    # see the plan first
pixi run python expts/preprocess/submit.py

# 3. watch it
pixi run python expts/preprocess/status.py --watch

# 4. verify, write the task lists, and publish
pixi run python expts/preprocess/finalize.py upload
```

Step 2 refuses to run from a dirty or unpushed tree — jobs clone the commit you
submit from, so commit first.

## What each piece is

| file | what it does |
|---|---|
| `download.py` | one resumable snapshot of the raw collection into `RAW_DIR` |
| `preprocess.py` | the job target: one database, rustler then embeddings |
| `submit.py` | sizes each job from `sizes.json` and submits it; also the paths |
| `status.py` | progress and ETA, measured in bytes |
| `finalize.py` | `verify`, `task-lists`, `upload` |
| `sizes.py` / `sizes.json` | expected output bytes per database |
| `rt-j-dbs.json` | the 475 databases rt-j trains on, carried forward |

Paths, embedder and batch size are the constants at the top of `submit.py`.
Everything else reads them from there, so changing where this writes is one
edit.

## Resuming, failures, preemption

**Re-run `submit.py`.** It is the resume: it skips databases whose output is
complete and those already queued or running, and submits the rest. That covers
a failed job, a job killed for running out of memory, a node going down, and a
sweep interrupted half-way.

A database is complete when its `meta.json` names an embedding file that exists.
Not when its directory exists — rustler writes its artifacts before the
embedding step, so a job preempted in between leaves a directory that looks
finished. `finalize.py verify` checks all eight expected files for all 639 and
refuses to upload if any is missing or empty, and `upload` runs `verify` itself.

Jobs run under the `il-lo` QOS, which is preemptible. Slurm requeues them and
the job target is idempotent, so preemption costs the work in flight and nothing
else.

If a database fails repeatedly, the log is at
`/dfs/user/ranjanr/slurm-logs/preprocess/<run_id>_<jobid>.out`. Out-of-memory is
the likely one: raise its tier in `TIERS`, or give it a one-off `Resources` by
hand.

## Why it is shaped this way

**One job per database, not a job array over shards.** The collection is
extremely lopsided: the median database preprocesses to ~43 MiB, the largest
(`join-tpch`) to 76 GiB, and 20 of the 639 are half the total bytes. A fixed
worker shape is therefore either too thin for the giants or wasteful for the
500-database tail, and round-robin sharding — which is what `rt.preprocess.many`
does — splits that distribution about as badly as it can be split. Sizing each
job from its expected output and letting slurm place them uses the scheduler for
what it is, and the makespan floor is `join-tpch` alone, which is why it gets 48
cpus rather than a worker's fair share.

This is only affordable because roach shares one clone per commit per node: the
first job on a node pays ~5 minutes for the clone, `pixi install` and the
rustler build, and every job after it starts in **1–2 seconds** (measured, 8
concurrent). At that startup, 639 separate jobs cost about 25 seconds of
makespan in total.

**Raw data is read from a local directory, not the Hub.** 639 jobs resolving
their own database would be 639 clients hammering the Hub in parallel; it
answers with HTTP 429 well before finishing. `download.py` fetches once.
`meta.json` still records the Hub spec (`stanford-star/the-join/<db>`) rather
than the local path — where the bytes were read and what they are are different
facts.

**Progress is bytes, not databases.** Counting databases would report the sweep
as nearly done while a quarter of the work remained. `status.py` weights each
database by what the previous build's output measured.

## The cluster, as measured

| node | cpus | memory | GPUs | local disk |
|---|---|---|---|---|
| hyperturing1 | 252 | 2011 G | 10 × RTX8000 48G | 5.7 T |
| hyperturing2 | 252 | 2011 G | 10 × RTX8000 48G | 752 G |
| turing1 | 80 | 754 G | 10 × 2080Ti 12G | 3.6 T |
| turing2 | 80 | 754 G | 10 × 2080Ti | 4.5 T |
| turing3 | 80 | 1448 G | 10 × 2080Ti | 11 T |

Jobs ask for one GPU each — needed only for the embedding step — so ~50 run at
once, using ~400–700 of the 716 cpus. Wall-clock limits and memory come from
`TIERS` in `submit.py`; memory per job is capped by the site's `MaxMemPerCPU` of
10700 M, so a tier that wants more memory has to ask for more cpus too.

`/dfs` measured ~700 MB/s buffered write and ~680 MB/s read from one node, so
writing the output there is not the bottleneck.

## Publishing

`finalize.py upload` verifies, regenerates `db-task-lists/`, pushes with
`upload_large_folder`, and then deletes the database directories the Hub has and
this build does not. That last step is what makes it a replacement: the previous
build has 650 databases, 11 of which were dropped from the raw collection for
overlapping RelBench and DBInfer sources, and a plain upload would leave them
there. Root files (`README.md`, `.gitattributes`) are not touched.

`db-task-lists/all.json`, `forecast.json` and `autocomplete.json` are derived
from the metas. `rt-j.json` is not derivable — it is a curated whitelist of 475
databases (126 excluded wholesale, none partially) — so the database list is
kept here in `rt-j-dbs.json` and the pairs are rebuilt against it. `task-lists`
prints any curated database missing from the build, which is how you notice the
whitelist has drifted from the collection.

## Regenerating `sizes.json`

```bash
pixi run python expts/preprocess/sizes.py
```

Reads file sizes from the published repo, downloading nothing. It is a
prediction, used for job sizing and progress weighting; a database that is not
in it is submitted at the largest tier, on the grounds that unknown is not the
same as small.
