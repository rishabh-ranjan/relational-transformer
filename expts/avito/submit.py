"""Submit the sampler probe. See [README.md](README.md)."""

from roach.slurm import Resources, submit

# The tasks to time. rel-avito/user-clicks is the one under suspicion; the
# rel-f1 line is the control -- a task whose fine-tuning run reaches step 1 in
# seconds, so a slow avito number means avito and not the machine.
TASKS = (
    ("rel-avito", "user-clicks"),
    # ("rel-f1", "driver-top3"),
)


def main() -> None:
    for db, task in TASKS:
        job = submit(
            "expts.avito.probe:main",
            args=dict(
                db=db,
                task=task,
                pre_dir="/dfs/user/ranjanr/share/stanford-star/relbench-preprocessed",
                num_batches=5,
                # 0 keeps the sampler in this process; the fine-tuning job's 16
                # would only hide a stall inside a worker.
                num_workers=0,
            ),
            # No GPU: the probe never builds the model, and blackwell1 is where
            # the job under investigation runs, so this times the same disk.
            resources=Resources(
                partition="il",
                account="infolab",
                qos="il-lo",
                time="02:00:00",
                gpus="0",
                cpus_per_task=16,
                ntasks=1,
                exclusive=False,
                mem="100000M",
                mem_per_gpu=None,
                constraint=None,
                nodelist="blackwell1",
            ),
            name=f"probe-{db}-{task}",
            repo_root="/lfs/hyperturing1/0/ranjanr/clones/rishabh-ranjan/relational-transformer",
            log_root="/dfs/user/ranjanr/slurm-logs/rishabh-ranjan/relational-transformer/expts/avito",
            clone_root="/lfs/local/0/roach_clones",
            secrets_dir="/dfs/user/ranjanr/.secrets",
        )
        print(f"  {db}/{task}: {job.log}")


if __name__ == "__main__":
    main()
