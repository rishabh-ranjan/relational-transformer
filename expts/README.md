# Experiments

One directory per experiment, laid out however that experiment wants. It has to
submit jobs and it has to be committed: jobs clone the commit you submit from.

## Layout

`fine_tune/submit.py` is the shape to copy: submit the entry point directly,
every argument spelled out at the call, a sweep as a loop around that call.
Keep a derived input (a curated task list, a subset) beside the file that uses
it.

## Editing submit.py

A submit script is edited, not configured — no arguments, no flags, no
`--dry-run`. Expect to change the file every submission, and **commit every
submission before submitting**: the job clones that commit. Rewrite the file
freely, git holds the variants.

Nothing in it is settled, and the resource plan least of all. **Work out which
gpus and which qos this submission should ask for at the moment you submit** —
following [Allocating a sweep](#allocating-a-sweep) against what the cluster
has free and what your own jobs already hold, plus whatever you have just been
told — and write that answer into the file. What the plan said last time is a record of a different
cluster and a different instruction, not a default to inherit: read it as one
more variant git is holding for you. The same goes for the task list, the
hyperparameters, and every other value in the call.

## Watch every job you submit

**A submission is not done when `sbatch` returns — it is done when you have
seen the jobs train.** Nothing alerts you, and a broken run holds its GPU for
the full wall clock while filling the shared filesystem. Reporting a submission
as finished before the checks below have passed is reporting work that was not
done.

**Waiting for the jobs to exit is not monitoring.** A command that blocks until
the queue is empty reports the end and nothing else: a run that stalled at hour
two, a job preempted back to the start, a log that stopped moving, all of it
arrives as one notification long after it could have been acted on. Monitoring
is a *repeated* check while the jobs are still running, and every round of it
ends in an answer to "is each job further along than last time?".

**Start one persistent monitor, not a chain of one-shot waits.** Arm a single
poll loop over the queue that runs for the life of the sweep and emits a line
per round; re-arming a timer at the end of every round is a fresh chance to
forget, and the first round nobody starts is the round the watch ends — which
looks exactly like a sweep that is fine. A stated interval with no live monitor
behind it is not a watch.

**Two monitors: one edge-triggered, one heartbeat.** A loop that speaks every
round has to be slow enough not to drown you, and a slow loop is exactly what
misses a slot that freed a minute after it looked. Split the two jobs: poll
often but **emit only on a change** — a tier or a card that has room, a pending
reason that will never start, a job that went from running back to the queue, a
reservation that moved or vanished — and run a second, quarter-hour loop that
prints one line whatever happened, so silence from the first is never mistaken
for a watcher that died. Keep the edge monitor's state between rounds and
compare against it; emitting the same condition twice is the same noise you
were avoiding.

**Make the monitor watch the tier counts, not just the jobs.** A high-tier slot
frees the instant any job of yours ends, which is far more often than any sane
interval, so a loop that only reports progress leaves you finding free `il`
slots by hand — or being told about them. Count the running jobs per qos
against [the budget](#allocating-a-sweep) every round and emit a distinct line
when a tier has room while anything of yours is pending; that line is a
[promotion to make now](#rebalance-while-it-runs), not a status update. A sweep
of short jobs turns tiers over every few minutes, so poll minutes, not
quarter-hours.

The loop has to end by itself, too: make it emit on the queue going empty, and
on the pending reasons below that mean a job will never start, not only on
progress. Silence has to mean "still going", never "the watcher stopped".

Every line of the budget is worth a counter, not just `il`: `il-interactive`'s
two, `il`'s b200 sub-cap, the b200 cards themselves, and the end time of any
reservation you are relying on. A card is only yours to take if its node is not
flagged `RESERVED` — a spare b200 under someone else's reservation pins the job
on `ReqNodeNotAvail` for good, so the monitor should read that flag rather than
`CfgTRES - AllocTRES` alone.

```bash
# The shape: a round line, and a PROMOTE line the moment a slot or card frees.
q=$(squeue -u $USER -h -o "%i %j %q %T %b %R" | grep -v dev-node)
il=$(echo "$q" | awk '$4=="RUNNING" && $3=="il"' | wc -l)
int=$(echo "$q" | awk '$4=="RUNNING" && $3=="il-interactive"' | wc -l)
ilb=$(echo "$q" | awk '$4=="RUNNING" && $3=="il" && $5 ~ /b200/' | wc -l)
pend=$(echo "$q" | grep -c PENDING)
bw=$(scontrol show node blackwell1)
bwfree=$(( 8 - $(echo "$bw" | grep -oE "b200=[0-9]+$" | cut -d= -f2) ))
echo "$bw" | grep -q "State=.*RESERVED" && bwfree=0      # not yours to take
scontrol show res <name> | grep -oE "EndTime=[^ ]+"      # empty = gone
[ $pend -gt 0 ] && [ $(( (10 - il) + (2 - int) )) -gt 0 ] && echo "PROMOTE: ..."
[ $pend -gt 0 ] && [ $bwfree -gt 0 ] && [ $ilb -lt 2 ] && echo "PROMOTE: b200 ..."
```

Check right after submitting, again once the runs are past startup — a few
minutes, longer with `compile=True`, and long enough that step lines must have
appeared by then — and then **on a fixed interval until every job has
finished**, an interval short enough to catch a stall in the same session it
happens. A sweep that runs for days is watched for days; a run breaks, stalls
or gets preempted long after it started training fine. Every time:

- read the log of **every** job, not a sample of them, and confirm each one
  reaches steps and its loss moves. A log that stops after the data stats is
  not yet evidence of anything;
- **compare against the previous round, not against zero.** The check that a
  job is alive is that its last line is *newer* than the one you saw last time.
  A run whose log has not moved since the previous round is stalled, whatever
  slurm says its state is;
- know **what each job's progress line costs**, so a gap between lines can be
  called normal or not: a training step is seconds, an eval pass over a whole
  test split is minutes to hours;
- `ls -laS` the log directory and `df -h` the output filesystem;
- **recount what is free and what you hold**, and move a pending job up if a
  better slot appeared — see
  [Rebalance while it runs](#rebalance-while-it-runs). A round that only looked
  at your own logs missed half of what changed;
- cancel what is broken, delete what it wrote, fix the cause, resubmit.

**A pending job is a job to diagnose, not a job to wait for.** `squeue -u
$USER -o "%.8i %.30j %.14q %.9T %R"` prints a reason beside every one, and only
some of them mean the queue is working:

- `Priority`, `Resources` — waiting on a card that is genuinely busy. Leave it,
  and keep watching: it still has to start.
- `ReqNodeNotAvail` — pinned by `nodelist` to a node with nothing free. It will
  sit there however long you leave it, because no other node can take it.
  Resubmit on a pool that has cards.
- `QOSMaxGRESPerUser`, `QOSMaxJobsPerUser`, `AssocMaxGRES` — over your own cap,
  usually behind your own other sweep. Move it to `il-lo` or wait the sweep
  out, deliberately.
- `PartitionTimeLimit`, `QOSMaxWallDurationPerJob` — the job asks for longer
  than the qos allows and will never run. Fix the plan.
- `launch failed requeued held`, `JobHeldUser`, `JobHeldAdmin` — dead. Cancel,
  fix, resubmit.

Everything below `Priority` in that list is a submission that did not happen.
Reporting a sweep as submitted while its jobs sit on one of those reasons is
reporting work that was not done.

Jobs that are still queued are not off the hook: they get the same checks once
they start. The watch ends only when every job you submitted has completed or
been cancelled — not when they have all been seen training once.

## Submitting

Jobs go through [`roach.slurm`](../src/roach/slurm/README.md) — read that first
for `submit()`, resource presets, and how a run survives preemption.

```python
from roach.slurm import BLACKWELL, submit

submit("expts.<name>.<module>:<function>", args={...}, resources=BLACKWELL,
       name=..., repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
```

- **Submit it, do not `srun` it**, a two-minute probe included. The repo lives
  on the submitting host's `/lfs`, which the compute node does not have; an
  `srun` from inside an allocation is a step of that job and inherits its
  limits; `Resources` carries the account, partition and constraint.
- **A probe is an experiment**: give it a directory under `expts/`, commit it,
  submit it as its own target. An uncommitted target cannot be run at all.
- **Write nowhere but the paths you pass the entry point.** The clone is shared
  by every job at that commit on a node and is read-only while jobs run; take an
  output root as an argument and put everything under it. See the
  [read-only section](../src/roach/slurm/README.md#the-clone-is-read-only).
- **`/tmp` is for a job's own scratch on its node, named after the run** — never
  for code, never for anything to be read afterwards (it is node-local, and a
  `#SBATCH -o /tmp/...` log vanishes).
- **Take the best slots available, absent an explicit instruction otherwise** —
  where "best" is what gets the sweep finished soonest, not what has the
  fastest card. See [Allocating a sweep](#allocating-a-sweep).

## Allocating a sweep

**The three qos tiers are a budget to spend, not a preference order over
cards.** Each cap is per user, counted across every job you have. Spend the
scarce, high-priority tiers on the fastest cards, then let the uncapped tier
take the rest. Fill in this order and stop when the sweep is placed:

| order | qos | budget | priority | wall | put it on |
| --- | --- | --- | --- | --- | --- |
| 1 | `il-interactive` | 2 gpus, any type | 1500 | 12h | **blackwell** — both of them |
| 2 | `il` | 10 gpus, **at most 2 b200** | 1000 | 7d | **2 blackwell, then ampere** for the other 8 |
| 3 | `il-lo` | uncapped (100) | 100 | 21d | ampere, and everything left over |

So a sweep's high-priority ceiling is **4 blackwell + 8 ampere = 12 jobs**;
job 13 onwards is `il-lo` and preemptible. `il`'s two b200 are a separate
sub-cap, not a slice of the ten — spending them costs 2 of the 10 as well.

**Priority buys the next card that frees, not a card.** A high-priority job
outranks every `il-lo` job in the queue, but it cannot take a card from a
running non-preemptible one — and `il` and `il-interactive` jobs are exactly
that. So before spending a tier on blackwell, check when a b200 will actually
free:

```
squeue -p il -h -t RUNNING -o "%u %b %M %l %q" | grep b200   # elapsed vs limit
```

Subtract elapsed from the limit for each of the 8 cards. If the soonest is
further out than the job itself would take on an ampere, **put the high tiers
on amperes instead** — a slower card now beats a faster card in four hours.
blackwell1 is one node of 8 shared with everyone, so this is the common case,
not the exception. Note it in `plan()`'s docstring when you do, with the
numbers you read.

Two other things that *are* reasons to place a job somewhere else:

- **Your own jobs holding the cap.** `il`'s ten count across all your sweeps,
  so a fine-tuning sweep already holding six leaves four here. Subtract what
  you hold before you spend.
- **A job that cannot resume having to fit the wall clock.** `il-interactive`
  is 12 hours and `il` is 7 days; a training run checkpoints and resumes
  through both, an eval run restarts from the top. Do not put a 15-hour eval
  in a 12-hour slot.

### A reservation is il-lo only

A node reserved for you is yours whatever tier the job asks for: nobody else
can take its cards, and nothing preempts you there. So a `il` or
`il-interactive` job on a reserved node **spends a capped, scarce slot on a
card it would have got for free** -- put `il-lo` jobs there and keep the high
tiers for the contended pool.

A reserved node is also out of general scheduling, so a job reaches it only
with `--reservation=<name>` (`Resources.reservation`), and a job that sets it
is confined to that reservation.

```
scontrol show res                     # name, nodes, end time
```

**Now**: `ranjanr_deadline` holds `ampere8` until 2026-08-13T00:00. `il-lo`
jobs only, and this line is worth rereading against `scontrol show res` -- a
reservation that has ended is a `--reservation` flag that fails the submission.

### Work it out fresh, every submission

**Never inherit the plan in the file.** `plan()` is a record of what the
cluster looked like and what you were told the last time someone submitted —
one more variant git is holding, not a default. Read the cluster, subtract what
you already hold, apply the table above, and write today's answer into the
file:

```
sacctmgr -np show qos format=Name,Priority,MaxTRESPU,MaxWall,Preempt
squeue -u $USER -h -o "%q %b %T" | sort | uniq -c   # what you already hold
sinfo -p il -N -o "%N %G %C %t"                     # what exists, what is down
squeue -p il -h -t RUNNING -o "%N %b" | sort | uniq -c
```

The `il` partition is one blackwell node and nine amperes, plus older cards:

| nodes | cards each |
| --- | --- |
| `ampere1`-`ampere9` | 8 x a100 |
| `blackwell1` | 8 x b200 |
| `hyperturing1`-`hyperturing2` | 10 x rtx8000 |
| `turing1`-`turing3` | 10 x 2080ti |
| `hyperion1`, `hyperion3` | 3-4 x titanxp |

A `Resources` for a b200 pins `nodelist="blackwell1"`, so those jobs can only
ever run on that one node — which is what the 2 + 2 budget above already
accounts for. Nothing pins the amperes.

### Rebalance while it runs

**An allocation is right when it is made and wrong an hour later.** What is
free changes without you: other people's jobs end, your own finish and hand
back your cap, and a preempted job of yours goes back to the queue. A sweep
left alone spends the rest of its life on `il-lo` behind everyone else.

So a monitoring round has **two** questions, not one. The first is
[per job](#watch-every-job-you-submit) — is each one further along than last
round? The second is about the cluster: *has what is free changed, and is there
a better slot for something of mine now?* Ask it every round, whether or not
anything of yours looks wrong.

Read the supply, not just your jobs:

```
squeue -u $USER -h -o "%q %b %T" | sort | uniq -c   # what you hold, by tier
squeue -u $USER -o "%.8i %.30j %.14q %.9T %R"       # your pending, with reasons
scontrol show node blackwell1 | grep -E "CfgTRES|AllocTRES"   # b200 free = Cfg - Alloc
sinfo -p il -N -o "%N %G %C %t"                     # ampere nodes, and what is down
squeue -p il -h -t RUNNING -o "%u %b %M %l %q"      # elapsed vs limit: when a card frees
```

**Any pending job of yours plus any free card is a move you have not made.**
That pair is the whole trigger; it does not need a reason beyond itself. The
most common way it appears is not someone else finishing — it is **your own job
completing and handing your cap back**, which is invisible unless you recount.
Short jobs make this constant: a sweep whose jobs take four minutes turns the
queue over many times inside a twenty-minute check.

What to do about it:

- **Recount every tier top down, as a fresh run of
  [Allocating a sweep](#allocating-a-sweep)** — not a patch to the placement you
  already have. Write the three rows out each round before deciding anything:

  | tier | cap | you hold | free |
  | --- | --- | --- | --- |
  | `il-interactive` | 2 gpus | ? | ? |
  | `il` | 10 gpus, ≤2 b200 | ? | ? |
  | `il-lo` | uncapped | — | — |

  Filling that in is what catches the tier you are not looking at. **The top
  tier is the one most often missed**: it is only 2 gpus, so it empties the
  moment its two jobs finish, and it empties silently. A round that moved a job
  up one tier while the tier above it sat unspent did the arithmetic on the
  wrong row.
- **Spend a freed tier on the jobs with the most left to do**, not on whatever
  is nearest to hand — the long pole finishes the sweep. A short job promoted
  onto a b200 buys minutes; the largest still-training run buys hours.
- **A freed slot is worth a card check, not an assumption.** `il-interactive`
  having room does not mean blackwell1 has a b200 free; read `AllocTRES` before
  spending it, and if there is none, re-ask the blackwell question below.
- **`CfgTRES - AllocTRES > 0` is not a card you can have.** A spare b200 may be
  held by a reservation, and a job pinned to `blackwell1` then sits on
  `ReqNodeNotAvail` forever rather than falling back to an ampere. So a
  blackwell submission is not done until you have seen it start: check its
  pending reason within a minute of submitting, and move it to an ampere the
  moment it reads `ReqNodeNotAvail`. This is the one move that can leave a
  *running* job worse off — it was cancelled to be promoted, and the promotion
  never ran.
- **If a tier has room while anything of yours is pending**, cancel the pending
  one and resubmit it into the tier that freed — a pending job has lost nothing
  by being moved.
- **Re-ask the blackwell question, both ways.** A high-tier job still pending on
  blackwell is a job you are paying priority for and getting nothing from: move
  it to an ampere. A high-tier job on an ampere when b200 cards have since freed
  is the same mistake mirrored: move it back. Read the remaining wall clocks, do
  not guess.
- **Move a *running* job only when what it loses is smaller than what it
  gains**: a run that checkpoints loses minutes, an eval that does not loses
  everything it has done, a pending job loses nothing. Prefer moving your
  youngest job when you need to free a slot of your own.
- **Move a checkpointing run by its `run_id`, never by starting it over.**
  `submit(..., run_id=<the cancelled run's id>)` reuses that run's `out_dir`, so
  the new job picks its `resume.pt` back up and the move costs at most
  `resume_save_mins`. Read the id off the submission's own output, check
  `resume.pt`'s mtime is recent, cancel, then submit. Confirm the new log says
  `resumed_from: ... step: <the step it was at>` — a job that reports
  `time_to_first_step` with no `resumed_from` has silently restarted from zero
  and has to be caught in that round, not later.
- **Only ever move your own jobs.** Other sessions' and other people's jobs
  count against the same per-user cap and are part of the arithmetic, but they
  are never yours to cancel.
- Leave the budget under-spent only deliberately, and say why.

**Set the interval from the job, not from habit.** The check has to be more
frequent than the thing it is watching for: a sweep of four-minute evals needs
minutes, a multi-day pretraining run does not. If your jobs are shorter than
your monitoring interval, you are guaranteed to miss every card that freed.

### Preemption is a state change to act on

`il-lo` is preemptible, and a preempted job is not a job that carries on. Watch
for it explicitly — a job whose `%M` elapsed went *backwards* between rounds, or
whose `%R` is a requeue reason, has restarted:

```
squeue -u $USER -h -o "%.8i %.30j %.9T %.10M %R"   # elapsed shrank = it restarted
sacct -u $USER -X --format=JobID,JobName%30,State,Elapsed --starttime now-1day
```

What it costs decides what to do. **A run that checkpoints resumes and has lost
minutes** — leave it, but count the restart. **An eval does not checkpoint: a
preemption restarts it from the top**, so a job that has been preempted once on
`il-lo` will lose everything again, and it is the first thing to promote when a
non-preemptible slot frees. Being preempted twice on the same tier is a
placement to change, not luck to retry.

## What every experiment owes

A README that someone who was not there can follow, in order, to run the work
again.

- **The README opens with the commands, in order** — what to run, what it costs,
  what to do when it stops. Numbers measured, and said to be measurements.
- **Every input is fetched by code in the directory**, not by a command someone
  typed once.
- **Anything derived, expensive and depended on is committed beside the code.**
  If regenerating it needs a resource that may be gone, keep the artifact, not
  the recipe alone.
`preprocess/` is the worked example.

## Commit what a re-run needs, delete the rest

- **Commit what a future re-run would want**: the submit script, a list that
  cannot be recomputed, a measurement that cost a sweep.
- **Delete everything else, wherever it landed** — scratch clones, probe logs
  and checkpoints, throwaway scripts, half-finished output. Nothing sweeps the
  shared filesystem, the node-local disks or `/tmp` for you.
- **The test is "would I read this next time, or write it again?"** Keep it only
  when re-deriving it is the expensive part.
- **Clean up when the question is answered**, not later.

## Teardown

"Tear down X" is one instruction, and it means all of this, without asking
which parts:

- **Kill every slurm job of X's** -- running, pending and requeued alike --
  and nothing else's.
- **Delete every file that exists only for X**: its submit script, its scan,
  any derived input only it reads. Deleted, not commented out; git holds it.
- **Take X out of everything that mentions it**: the experiment's README, a
  shared workspace or results script, a sweep list that still names it.
- **Delete X's scratch**: clones, logs and checkpoints under
  `/lfs`, `/tmp` and `/dfs/user/<you>` alike.
- **Keep the finding, not the machinery** -- a decision X settled belongs in
  the experiment's README, in a sentence.
- **Commit and push it as one change.** A half-torn-down experiment reads as a
  live one.

Whatever the answer was, the runs are over: an arm that keeps training after
its question is answered is spending someone else's cards.

## What is specific to this repo

- **Nothing to build past `pixi install`.** The rustler sampler is a compiled
  extension, but the project is an editable dependency of its own environment,
  so building the clone's environment builds it. Pass `setup=` only for work
  that is not part of the environment (fetching a model, say).
- **Data is a local directory.** Nothing is fetched from the Hub at run time
  (see [docs/downloads.md](../docs/downloads.md)); point `pre_dir` at a path
  every node can read, and expect a job's first minutes to go on populating it
  into the page cache.
- **Entry points take every argument explicitly.** `rt.train._train` has no
  defaults; [`examples/`](../examples/) has the released values to start from.
