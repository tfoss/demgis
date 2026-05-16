# Ocean tile unification — work-item beads

Decomposes phase **E2** of [`../MIGRATION_PLAN_DRAFT.md`](../MIGRATION_PLAN_DRAFT.md). Design spec is [`../OCEAN_TILE_GUIDELINES.md`](../OCEAN_TILE_GUIDELINES.md). Each bead is a self-contained work item that an implementer (human or agent) can pick up cold: read the bead + the spec sections it links to, and ship.

## Beads

| # | Bead | Depends on |
|---|---|---|
| 01 | [Precomputes + STRtree spatial index](01_precomputes_and_strtree.md) | — |
| 02 | [Geometric primitives (tangents + coast tracing + sector polygon)](02_geometric_primitives.md) | — |
| 03 | [OceanExtension schema migration](03_schema_migration.md) | — |
| 04 | [`compute_ocean_extension` orchestrator](04_compute_ocean_extension.md) | 01, 02, 03 |
| 05 | [QC harness](05_qc_harness.md) | 04 |
| 06 | [Pilot 1 — Japan+Korea (regression test)](06_pilot_japan_korea.md) | 01–05 |
| 07 | [Pilot 2 — Sri Lanka (single-pair clean validation)](07_pilot_sri_lanka.md) | 01–06 |
| 08 | [Pilot 3 — Great Britain (multi-neighbour sector union)](08_pilot_great_britain.md) | 01–07 |
| 09 | [Pilot 4 — Cuba+Caribbean (multi-pair, three-way junction)](09_pilot_cuba_caribbean.md) | 01–08 |
| 10 | [Inland water bodies (lakes + inland seas)](10_inland_water_bodies.md) | 03, 04, 05 (Cat 2B); independent for Cat 2A |
| 11 | [Dovetail-split STL generation (feasibility investigation)](11_dovetail_split_investigation.md) | — |
| 12 | [Country STL splitting — components + dovetail](12_country_split.md) | 11 (print-validated parameters), 01–04 |
| 13 | [Country-name text recessed into STL back](13_country_name_back_label.md) | 12 (per-piece labels) |
| 14 | [Label strict containment (fix letters falling outside country)](14_label_strict_containment.md) | 13 |

Pilot 5 (Indonesia / Malaysia / PNG) is deferred until Phase C (LCC migration) lands — no bead yet.

## Dependency graph

```
01 ──┐
02 ──┼──► 04 ──► 05 ──► 06 ──► 07 ──► 08 ──► 09
03 ──┘
```

Beads 01, 02, 03 are independent leaves and can be worked in parallel. 04 composes them. 05 layers QC on the orchestrator's output. The four pilots run sequentially because each one validates a property the next one assumes is working: 06 = regression, 07 = single-pair, 08 = multi-neighbour union, 09 = three-way junction.

## How to use

When picking up a bead:
1. Read the bead file end-to-end.
2. Read the linked sections of `MIGRATION_PLAN_DRAFT.md §5` and `OCEAN_TILE_GUIDELINES.md` — the bead points at specific sections rather than restating them.
3. Resolve the bead's "Open questions" before starting, or flag them to the user.
4. Honour "Out of scope" — the boundaries between beads exist to keep PRs reviewable.

When a bead lands, update its "Status" line in the bead file (Not started → In progress → Done + commit ref).
