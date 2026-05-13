# Bead 11: Dovetail-split STL generation (feasibility investigation)

**One-line goal:** Investigate whether automated multi-piece splitting with dovetail joints can be baked into the STL generation pipeline, replacing the current manual BambuStudio post-processing step for oversized countries (US, China, Brazil, India, eventually Russia).

**Status:** Done — feasibility report + PoC delivered (see `docs/dovetail_split_feasibility.md` and `bin/poc_dovetail_split.py`). Verdict: implement; 3–5 engineer-days. One open print-test required to verify FDM layer-line load handling.

## Context

Countries whose extent exceeds the print bed (220 mm at `GLOBAL_XY_SCALE=0.33` → 667 km in CRS) currently get split in BambuStudio's slicer using its "dovetail" cutting option. The slicer auto-generates one large dovetail joint at the cut line, which gives a tight friction fit when the two pieces are assembled.

The pattern works but lives outside the STL pipeline: every oversized country requires a manual slicer pass. Moving the split into the STL generator would let `make_country_group.py --group USA` directly emit `USA_west.stl` + `USA_east.stl` (or more pieces, as needed) with the dovetail joints already cut. Benefits:
- Reproducible output without per-country slicer steps
- Joint parameters tunable in code rather than in the slicer GUI
- Joints survive STL re-generation (slicer-side edits are lost on regenerate)
- Compatible with the existing per-country regeneration workflow

This bead is **investigation only**: produce a feasibility report and an implementation-complexity estimate. Actual implementation is a follow-up bead.

## Inputs

- `/workspace/CLAUDE.md` — print bed size (220 mm), pipeline overview, `GLOBAL_XY_SCALE`.
- BambuStudio's documentation / source for its dovetail cut implementation (web search; or examine an STL exported from BambuStudio to reverse-engineer the joint geometry).
- Existing pipeline mesh code: `make_all_sa_with_vector_clip.py`, `country_to_solid_stl_with_star.py`, the bridge/cutout mechanisms in `make_country_group.py`.
- trimesh boolean ops + `manifold3d` backend (already a dependency, used for capital-star cutting and ocean masking).

## Investigation areas (in priority order)

1. **BambuStudio's dovetail joint geometry.** Document its shape, parameters, and joint kinematics. Either (a) read BambuStudio's source or docs, or (b) run a known small STL through BambuStudio's dovetail cut and inspect the output to reverse-engineer the joint shape. Output: a small reference diagram + parameter list (joint width, depth, angle, tab count per cut).
2. **Cut-plane decision.** For a country that exceeds the print bed, how is the cut line chosen?
   - Axis-aligned midline (X or Y, whichever extent is larger)?
   - Optimal split that minimises piece count (recursive halving)?
   - User-specified per country (e.g. `group.split_plan: {"USA": ["x=-100°"]}`)?
   - Avoid cutting through populated/landmark regions (politically sensitive — split USA at the Continental Divide rather than through Chicago)?
3. **Dovetail-construction approach.** Two implementation paths:
   - **Boolean subtract a dovetail-shaped slot** from one piece, **boolean add a matching tab** to the other. Symmetric; trimesh `difference` + `union`.
   - **Cut with an interlocking plane** (a flat plane with dovetail-shaped extrusion built in). One boolean per piece.
   The second is simpler but constrains the joint geometry.
4. **Joint parameters by piece size.** A 220 mm × 220 mm piece needs a different joint scale than a 100 mm × 50 mm piece. Investigate scaling rules — fixed dovetail size? proportional to cut-line length? user-overridable per cut?
5. **Print orientation implications.** Dovetails need to be oriented so the FDM layer lines align with the joint's force direction (i.e. the dovetail tab pulls perpendicular to layers, not parallel — otherwise the tab snaps). Verify this is preserved given the existing pipeline's bed-aligned orientation.
6. **Computational cost.** Boolean ops on 2M-face country meshes are non-trivial. Estimate runtime for cutting a US-sized mesh into 3 pieces with dovetail joints. May want simplification before the cut.
7. **UX and configuration.** How does the user specify split intent? Per-group YAML? A `--split` flag? Auto-detect from extent?

## Deliverables

A single Markdown report `docs/dovetail_split_feasibility.md` containing:

1. BambuStudio dovetail joint reference (diagram + parameters).
2. Feasibility verdict: **yes / yes-with-caveats / no, because…** for each of the seven investigation areas.
3. Recommended implementation approach (one of: boolean cut + add, plane sweep, hybrid).
4. Implementation-complexity estimate in days. Break out: cut-plane logic, joint geometry, boolean ops, integration with driver, tests.
5. Recommended UX: where the split intent lives (CountryGroup config? driver flag? both?).
6. One **proof-of-concept STL** for a single test case: take a 400 mm × 200 mm mesh (synthetic or a downscaled USA), split it at the midline with a single dovetail joint, save as two STLs. Visual / printability comparison against BambuStudio's output on the same input.
7. Risks and open questions for the follow-up implementation bead.

## Acceptance criteria

- Report is concrete enough that a follow-up implementation bead can be scoped directly from it (named functions, file locations, parameter defaults).
- Proof-of-concept STL pair imports cleanly into BambuStudio and demonstrates the joint geometry.
- Feasibility verdict is unambiguous — either "implement" or "don't, because <specific reason>".
- Complexity estimate is honest (range, with the load-bearing unknowns named).

## Dependencies

None. This bead can run in parallel with any of beads 06–10. The output may inform the global scaling discussion (bead TBD), since dovetail-split feasibility affects how aggressively we can scale before split-count becomes painful.

## Out of scope

- Implementation. This bead produces a report + a single PoC STL, not a working pipeline.
- Joint geometries other than dovetail (puzzle, mortise-and-tenon, magnetic). If the report recommends a non-dovetail joint, that's a finding but not a deliverable.
- Multi-axis splits (X *and* Y on the same country — e.g. quartering Russia). 1D splits only for the feasibility test.
- Splitting at terrain features (mountain ranges, rivers) rather than straight planes. Mention as a future direction but don't investigate.
- Curved cut lines. Straight planes only.

## Open questions

1. **Is BambuStudio's joint geometry proprietary?** If reverse-engineering from output is required, the joint we replicate may not be identical. Acceptable if our independent joint fits at the same tolerance.
2. **Does the existing pipeline's mesh fix-up step (`fix_normals`, `manifold3d`) survive a dovetail cut?** Boolean ops are flakey on near-degenerate geometry; the dovetail's small features may trigger failures.
3. **Where does this run in the pipeline?** Pre-simplification (clean mesh, slow) or post-simplification (fast, but the joint may be coarse)? The PoC should test both.
4. **Should split metadata be recorded in `alignment.json`** so the slicer / assembly step knows which STLs go together?
5. **Are dovetails always two-piece, or do we need three-piece junctions** (where three pieces meet, e.g. a USA quartered into NE/NW/SE/SW)? Out of scope for this bead, but the joint geometry should not preclude a future three-piece variant.
