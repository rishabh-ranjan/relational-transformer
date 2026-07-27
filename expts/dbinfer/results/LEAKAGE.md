# Does `rdblearn_tabicl` leak?

Motivation: RT-J losing to DFS+TabICL by this margin is surprising enough that the
baseline deserves to be treated as the suspect, not the reference.

**Conclusion: no leak that could explain the gap.** Three independent tests below.

## 1. No single feature is near-perfectly predictive

A leaked label shows up as one feature with AUROC ~1.0. Per-feature AUROC against the
true test labels, taking `max(a, 1-a)` so an inverted feature cannot hide:

| task | features | n | **max single-feature AUROC** | >0.95 | >0.99 |
|---|---|---|---|---|---|
| `diginetica/ctr` | 25 | 6,616 | 0.6620 | 0 | 0 |
| `retailrocket/cvr` | 38 | 9,997 | 0.7432 | 0 | 0 |
| `stackexchange/churn` | 44 | 105,612 | 0.8130 | 0 | 0 |
| `stackexchange/upvote` | 34 | 38,588 | 0.8245 | 0 | 0 |

Not one feature above 0.95 anywhere.

## 2. The model scores about what its features are worth

If the label were leaking, the model would land near the leaking feature's AUROC
(~0.99). Instead it lands slightly above its *best* feature -- the signature of an
ordinary tabular model combining many weak-to-moderate features:

| task | best single feature | model (fulltest, ctx=8192) | lift |
|---|---|---|---|
| `ctr` | 0.6620 | 0.6836 | +0.02 |
| `cvr` | 0.7432 | 0.7880 | +0.04 |
| `churn` | 0.8130 | 0.8446 | +0.03 |
| `upvote` | 0.8245 | 0.8525 | +0.03 |

A +0.02..0.04 lift over the best of 25-44 features is what a good tabular model does.
It is not what leakage looks like.

## 3. Features respect each row's cutoff

DFS computed once, ignoring the per-row cutoff, would be a temporal leak that raises
many features mildly rather than any one to 1.0 -- so test 1 would not catch it.
`stackexchange/churn` has every one of its 88,164 val entities also present in test,
under two different cutoffs (val 2021-01-01, test 2023-01-01):

```
entities in both val and test              : 88,164
identical feature vectors across cutoffs   : 0.0000
mean fraction of feature entries differing : 0.3632
```

Zero rows carry the same vector across the two cutoffs. The featurizer is
cutoff-aware. (`include_cutoff_time=False` -- strict `<` -- is pinned in
`rel2tab/featurizers/rdblearn_featurizer.py` so an upstream default flip cannot
silently change it.)

## One real but immaterial defect

12 of 105,612 `churn` test rows belong to users whose `Users.CreationDate` is *after*
the test cutoff. Those users have no pre-cutoff history at all, yet 43% of their
feature entries are non-zero (against 75% for other rows), and none is entirely zero
-- consistent with DFS including the seed entity's own attributes, which for these
users are themselves post-cutoff facts.

It is a leak in principle. It is 0.011% of rows, it cannot move AUROC at the third
decimal, and it does not touch the other three tasks' conclusions.

## What this means

The baseline's advantage is real. It is not an artifact to be explained away, and
the RT-vs-baseline gap has to be explained on the modelling side -- see `AUDIT.md`,
which locates it in 4DBInfer supplying ~10x less in-context supervision per entity
than RelBench, which DFS aggregates are indifferent to.

---

# Part 2: does the *data source* differ between the two methods?

A sharper version of the leakage question, since the two methods do not read the same
files. `rdblearn`/`fastdfs` reads the original 4DBInfer tars via
`RDBDataset.from_4dbinfer`; RT reads the rustler-preprocessed `stanford-star/dbinfer`
port. Any drift between those is indistinguishable from leakage in the results.

## The danger: raw parquets carry undeclared columns

The raw files hold columns `metadata.yaml` never declares, and they are exactly the
incriminating ones:

| table | undeclared columns in the raw parquet |
|---|---|
| `Posts` | **`Score`**, `ViewCount`, `AnswerCount`, `CommentCount`, `FavoriteCount`, `ClosedDate`, `Tags`, … (13) |
| `Users` | **`Reputation`, `UpVotes`, `DownVotes`, `Views`**, `AccountId`, `DisplayName`, … (8) |
| `Comments` | **`Score`**, `ContentLicense`, `UserDisplayName` |

`stackexchange/upvote`'s label is derived from `Posts.Score`. `churn` is user activity,
which `Users.UpVotes`/`Reputation` encode almost directly. Had `fastdfs` read the
parquets as-is, the whole comparison would be void.

## It does not. Verified by running the adapter, not by reading it

`DBInferAdapter.load()` yields exactly the metadata-declared set:

```
Posts:    AcceptedAnswerId, Body, CreationDate, Id, LastEditorUserId,
          OwnerUserId, ParentId, PostTypeId, Title
Users:    AboutMe, CreationDate, Id, Location
Comments: CreationDate, Id, PostId, Text, UserId
```

No `Score`, no `Reputation`, no `UpVotes`, no `DownVotes`, no `ViewCount`.

Exhaustively, over all 21 tables of all 3 databases:

| check | result |
|---|---|
| undeclared columns reaching the adapter | **0 / 21 tables** |
| adapter column set vs port column set | **identical, 21 / 21** |
| row counts, raw vs port | **identical, 21 / 21** |
| non-key value multisets, adapter vs port | **identical** (one false positive, below) |

Row order and key encoding differ between source and port, so the value check compares
permutation-invariant multiset signatures rather than positions.

The single flag, `retailrocket/ItemAvailability.available`, was the checker's fault:
raw stores it as str `'0'/'1'`, the port as float64 `0.0/1.0`, counts identical
(863,086 / 640,553). Metadata declares the column `float`, so the port's encoding is
the correct one.

## The asymmetry that does exist runs *against* RT

`rdblearn/estimator.py:298` applies `FilterColumn(drop_dtypes=["text"])`. **Every text
column is dropped before DFS ever runs.** RT embeds all of them with
`all-MiniLM-L12-v2` and carries them in context.

| database | text columns rdblearn drops, RT sees |
|---|---|
| `stackexchange` | 9 of 48 declared -- `Posts.{Body,Title}`, `Comments.Text`, `PostHistory.{Comment,Text}`, `Users.{AboutMe,Location}`, `Badges.Name`, `Tag.TagName` |
| `retailrocket` | 1 -- `ItemProperty.value` |
| `diginetica` | 0 |

### Materialized dimension tables are *not* an RT-only artifact

Several 4DBInfer foreign keys point at tables that do not exist in the raw data
(`View.visitorid` with no `Visitor` table, and so on), so any valid relational schema
has to materialize them. The port does it at port time; **rdblearn does the same thing
at pipeline time** -- `HandleDummyTable()` is the first step of
`rdblearn/estimator.py`, "Create dummy tables for missing primary key references",
followed by `FillMissingPrimaryKey()` which expands PK tables to cover FK-referenced
values. Same tables, same rows, different moment. No asymmetry.

They also cost RT **nothing** in context budget. `rustler/src/pre.rs:592` skips the
primary-key column when emitting cells, and foreign-key columns are consumed to build
the p2f adjacency rather than emitted as content -- so a key-only table has no
emitted columns and its nodes contribute **zero cells**:

| database | materialized nodes | share of db nodes | cells contributed |
|---|---|---|---|
| `diginetica` | 706,016 (`Orders`, `Session`, `Token`, `User`) | 0.73% | **0** |
| `retailrocket` | 1,874,368 (`Item`, `Visitor`) | 7.53% | **0** |
| `stackexchange` | none | 0% | 0 |

They are pure routing nodes, and load-bearing at that: without a shared `Item` node,
two `View` rows on the same item have no path between them at all. The only budget
they can touch is `bfs_width` -- a materialized node selected at a BFS level occupies
one of that level's slots while yielding no cells -- which is a traversal
inefficiency, not context dilution.

So on the two axes where the methods differ, RT has *more* information, not less, and
still loses. That direction matters: it means the gap cannot be explained away as the
baseline seeing something RT does not.

## Angles still open

* **DFS depth vs walk depth.** `max_depth=2`: rdblearn aggregates 2 hops *exhaustively*;
  RT reaches further but only along sampled walks. Exhaustive-shallow versus
  sampled-deep is a real asymmetry in access pattern, not in content, and on
  low-label-density tasks the exhaustive side is favoured.
* **Semantic-type hints.** The adapter assigns `category_t`/`float_t`/`datetime_t` from
  metadata; rustler infers its own. A column read as category by one and numeric by the
  other carries the same information in a different representation.
* **`dbinfer-diginetica` still carries its `purchase` task table** in the preprocessed
  graph, consuming RT's context budget on `ctr` with no analogue on the rdblearn side.
