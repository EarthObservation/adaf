# STONE repository audit

**Audit date:** 2026-08-07 (Europe/Ljubljana)  
**Repository root:** `C:\Users\ncoz\GitHub\adaf-stone`  
**Audit mode:** static, read-only inspection; notebook cells were not executed

Finding labels used throughout:

- **Confirmed**: directly supported by repository content or Git output.
- **Likely interpretation**: a reasoned interpretation that needs runtime or researcher confirmation.
- **Unresolved**: the repository does not contain enough evidence to decide.

Notebook references use **zero-based indices across all notebook cells**, including Markdown, raw, and code cells. This is stable even when execution counts are missing or out of order.

## 1. Executive summary

This is a valid checkout of the expected ADAF repository, not an unrelated fork. The expected remote is configured as `origin`, GitHub reports `main` as its default branch, and the current `stone` commit (`815df521...`) is the exact merge base with current upstream `main` (`a44acdd...`). The committed STONE work was merged upstream in pull request #9; this checkout is now 15 commits behind upstream, whose later file changes are limited to `README.md`, `TRAIL2025.md`, and removal of four vendored wheel paths. The scientific work most in need of preservation is instead in six unstaged edits and 32 untracked files.

The repository contains a usable ADAF-derived inference core, two competing STONE dataset-grid implementations, a custom semantic-segmentation dataset, HRNet retraining work, an alternative ViT segmentation experiment, three evaluation families, and model-assisted reference-data refinement. It is not currently a reproducible experiment package: data, weights, exact environment, seeds, dataset manifests, and resolved configurations are external or absent, and many notebooks embed machine-specific paths.

Most consequential confirmed findings:

1. **The current dataset split is not auditable from the repository alone.** The data are absent and no manifest records sample IDs, geometries, source rasters, checksums, or split membership. The three builders use different rules. `adaf/grid_patches.py` has a boundary rule that can classify a tile as training even when it crosses the outer validation/test boundary, and its worktree version filters test geometry before reprojecting it to the raster CRS (`build_learning_dataset_grid`, lines 207-263).
2. **Mask and class assumptions conflict.** `create_segmentation_mask()` always writes a three-band mask and assumes a one-band image when constructing it (`adaf/create_patches.py`, lines 208-265). The saved e4MSTP notebook error confirms failure on a multiband source. Current barrow training uses band index 0, while `STONE_retrain.ipynb` uses band index 1 for barrow.
3. **Inference aggregation can change scientific results.** Object-detection boxes are merged only within each prediction text file, not across overlapping tiles (`adaf/adaf_inference.py::object_detection_vectors`). Semantic-segmentation polygons are dissolved across every label without grouping by label, so overlapping multi-class output can lose class identity (`semantic_segmentation_vectors`, lines 211-224).
4. **The alternative centroid metric is greedy and order-dependent.** It removes elements from lists while iterating and uses strict centroid containment with an IoU threshold of zero in all saved comparison notebooks (`adaf/evaluate.py::compute_iou_metric_centroid`; `new_eval_measure ...`, cell 9). The saved TP/FP/FN tables are evidence of past runs, but they are not traceable to committed data, weights, configuration, or a code version.
5. **Notebook state is materially stale.** Twelve notebooks retain saved errors, interrupted runs, or obviously out-of-order execution; several large notebooks store 1-2.6 MB of output. `ADAF_trainning_samples.ipynb` calls the current four-argument `create_patches_main()` with three arguments (cell 9), and `ADAF_notebook.ipynb` leaves `out_dir=None` before `main_routine()` calls `Path(inp.out_dir)`.

**Overall assessment:** preserve the worktree immediately, establish one authoritative dataset/split manifest and one evaluation specification before interpreting model rankings, then consolidate code. Cleanup before those decisions risks losing provenance or silently changing the scientific question.

## 2. Audit scope, date, current branch, commits, and limitations

### Scope and baseline

| Item | Finding |
|---|---|
| Repository root | `C:\Users\ncoz\GitHub\adaf-stone` |
| Current branch | `stone`, tracking `origin/stone` |
| Current commit | `815df52122605dbf73aeccf3b35dbb4af29e98bc` - “Add model training notebook and dataset script” (2026-01-23) |
| Current upstream default | **Confirmed `main`** by `git ls-remote --symref` |
| Current upstream `main` | `a44acdd678773666458ea50f544b5d2292bcef51` (2026-04-17) |
| Remote | `origin https://github.com/EarthObservation/adaf.git` for fetch and push |
| Audit date | 2026-08-07 |

### What was inspected

- Git status, refs, history, merge bases, tree sizes, ignored files, object connectivity, and three-way diffs.
- All 35 non-checkpoint notebooks as JSON, without execution.
- All 19 Python files, the launcher, README, ignore/lint configuration, IDE metadata, wheel metadata, media/binary classifications, and the absolute-path list.
- Notebook source cells, imports, definitions, paths, execution order, saved outputs, and errors.
- The committed STONE lineage starting at the pre-branch point `925493e...` and later upstream history.

### Limitations

- **Confirmed:** no repository data rasters/vectors, model weights, database contents, or generated GeoPackages are present, so sample duplication, actual spatial leakage, CRS equality, class counts, and reported values could not be independently recomputed.
- **Confirmed:** there is no runnable Python interpreter on the audit PATH and package installation was prohibited. Python syntax/import checks and unit tests were therefore not run. `git diff --check` and `git fsck --connectivity-only` were run instead.
- **Confirmed:** PDF extraction/rendering utilities were unavailable and installation was prohibited. `ADAF_manual_v1.1.pdf` was classified by path, size, and Git history, but its page content could not be compared with current code. This is an explicit documentation-review gap.
- **Confirmed:** the vendored AiTLAS wheel metadata was read without installation. Runtime behavior internal to AiTLAS (including exact transforms, metric aggregation, checkpoint semantics, and model compatibility) remains partly unresolved.
- **Confirmed:** no notebook cell, training, inference, download, database query, or write was executed.
- **Confirmed:** `nbdime` was not available. Notebook comparison used parsed code-cell sources while ignoring outputs and metadata.

## 3. Git state and upstream baseline

### Working tree

At audit start there were no staged files, six unstaged tracked files, 32 untracked non-ignored files, and 31 ignored files.

**Unstaged tracked files:**

- `ADAF_trainning_samples.ipynb`
- `ADAF_widget.ipynb` (saved output only)
- `STONE_create_learning_dataset.ipynb` (Markdown encoding and saved-output changes; no substantive code-cell change)
- `adaf/adaf_utils.py`
- `adaf/grid_patches.py`
- `train_and_evaluate_semantic_segmentation.ipynb`

`git diff --stat` at audit start reported 369 insertions and 81 deletions across those six files. The semantic-training notebook accounts for most of that because it was expanded and executed locally. `adaf/adaf_utils.py` contains a substantive uncommitted fix for `nodata=None` and TIFF predictor selection. `adaf/grid_patches.py` changes split names, test-negative inclusion, grid filtering, defaults, and example paths.

**Untracked files:**

```text
.vscode/settings.json
Object_Detection_Evaluation.ipynb
STONE-data_loader.ipynb
STONE_create_learning_dataset-e4MSTP.ipynb
STONE_retrain.ipynb
STONE_retrain_OLD.ipynb
STONE_retrain_OLD2.ipynb
VIT_e4mstp.ipynb
VIT_e4mstp_sample_display_added.ipynb
_test_crete_patches.py
_test_inference.py
adaf/eval.py
adaf/evaluate.py
adaf/stone_inference.py
echoes_paths_s2q.txt
echoes_s2patches.ipynb
echoes_vectors.ipynb
evaluate.py
learning_dataset_grid.ipynb
new_eval_measure adaf_bih_0-256px.ipynb
new_eval_measure adaf_bih_0-512px.ipynb
new_eval_measure adaf_retrained_2-256px.ipynb
new_eval_measure irish.ipynb
new_eval_measure retrained_1.ipynb
new_eval_measure_template.ipynb
reference_data_refinment.ipynb
test_vit_inferenc.ipynb
torchgeo_001_multimodal_sample.ipynb
torchgeo_002_datasets.ipynb
train-eval_BIH_v5_run4.ipynb
vision-transformers-for-segmentation.ipynb
vit_adaf_standalone_inference.py
```

Ignored material includes `.idea/`, nine notebook checkpoints, `Untitled.ipynb`, and Python bytecode. `Untitled.ipynb` is scientifically relevant despite being ignored and is included below.

### Branches and history

| Ref | Commit | Relevance |
|---|---|---|
| `stone`, `origin/stone` | `815df52` | Current checkout; +0/-0 relative to its remote |
| `main`, `origin/main`, `origin/HEAD` | `a44acdd` | Verified current upstream default/tip |
| `origin/demo/adaf-lite` | `4face36` | Alternative upstream effort to remove AiTLAS; not merged into `main` |
| `origin/unit_test` | `500b38f` | Old remote branch; no current test suite in this checkout |

The current `stone` commit is an ancestor of upstream `main`: `git rev-list --left-right --count HEAD...origin/main` returned `0 15`. The merge base is `815df52`, so there are no committed local-only changes *after* the official current merge base. Pull request merge `a9bf715` incorporated the STONE branch into upstream `main` on 2026-04-14.

For provenance, commit `925493e0a4e5c9d8874ec8c1784d8cc7b54747e6` (2025-08-20) is the last common mainline commit before the October 2025 STONE branch sequence shown by the graph. Comparing that point to `815df52` identifies 18 committed STONE-touched files: 12 modified ADAF files and six new STONE assets (`STONE_create_learning_dataset.ipynb`, `STONE_train_new_model.ipynb`, `stone_dataset.py`, `adaf/create_singleclass_dataset.py`, `adaf/grid_patches.py`, and `adaf/media/create_singleclass_dataset.png`). These changes are no longer private: they exist in upstream history.

### Size and large files

- Working tree excluding `.git`: **97,102,998 bytes** (~92.6 MiB).
- Git object storage: **91.60 MiB packed + 10.70 MiB loose**.
- Largest tracked files are three GDAL wheels at ~25.75 MB each. The Python 3.8 GDAL wheel is duplicated byte-for-byte under `_wheels/` and `installation/`.
- Five evaluation-result notebooks are ~1.10 MB each, `new_eval_measure_template.ipynb` is 2.49 MB, `torchgeo_002_datasets.ipynb` is 2.60 MB, and `VIT_e4mstp.ipynb` is 1.36 MB, primarily because of saved output.
- `echoes_paths_s2q.txt` is an untracked list of 1,671 unique absolute Windows paths to GeoJSON files.
- No model checkpoint or raw geospatial dataset is committed. This keeps Git size manageable but leaves experiments dependent on external state.

`git fsck --connectivity-only` succeeded but reported dangling objects/commits, which is normal evidence of prior rewritten/unreferenced local Git activity and not repository corruption. `git diff --check` found one trailing space in the existing uncommitted `adaf/adaf_utils.py` edit.

## 4. Three-way comparison with upstream ADAF

The official merge base (`M`) is the current `stone` HEAD. Therefore:

- `M -> local HEAD`: no committed changes;
- `M -> upstream main`: later upstream changes;
- `local HEAD -> working tree`: six tracked edits plus 32 untracked files.

The older provenance comparison `925493e -> 815df52` is used only to identify the STONE branch's committed contribution; it is not substituted for the official merge base.

| File or area | Local STONE change | Later upstream change | Possible conflict | Suggested response |
|---|---|---|---|---|
| STONE dataset/training additions | Added on `stone` and now in upstream history | None after `815df52` | No current text conflict | Treat current committed versions as upstream-integrated, then separately preserve worktree experiments |
| Core ADAF inference/visualisation/patch code | Modified in the STONE commit chain; `adaf_utils.py` now has an additional unstaged fix | None after merge | Low immediate merge conflict; scientific behavior still diverges from older ADAF | Commit reviewed fixes only after regression tests against ADAF behavior |
| `adaf/grid_patches.py` | Added by STONE; extensively modified in worktree | None | No upstream text conflict, but competing with `create_singleclass_dataset.py` | Choose authoritative split semantics before merge/refactor |
| ADAF notebooks | Kernel metadata changed in commits; three have local content/output changes | No later upstream notebook edits | Low Git conflict, high hidden-state/documentation risk | Preserve and normalize only after extracting configurations and outputs |
| `README.md` | STONE acknowledgement and installation edits are in branch history | Later upstream revised installation, citation, DOI, and ARCHON links | Local branch is missing current documentation | Rebase/merge only after worktree preservation; review launcher/environment-name mismatch |
| `TRAIL2025.md` | Absent locally | Added and revised upstream | No path collision | Bring in with upstream update |
| `_wheels/*` | Retained locally | Entire directory deleted upstream | Deletion versus retention policy | Confirm offline-install needs; align to upstream or external artifact storage after decision |
| `installation/GDAL-3.4.3-cp38...whl` | Retained locally | Deleted upstream | Same as above | Do not delete until supported Python/GDAL strategy is decided |
| `installation/aitlas...whl` | Retained | Not deleted on upstream `main` | None | Preserve until dependency replacement/versioning decision |
| 32 untracked files | STONE experiments and local tooling | No matching paths upstream | They will not be protected by an upstream update | Snapshot/classify before any branch operation |

## 5. Repository inventory

### 5.1 Python modules and executable files

| File | Origin | Purpose | Status | Inputs | Outputs | Overlaps with | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|---|
| `adaf/__init__.py` | ADAF | Package marker | Empty, active | None | None | None | Keep | High |
| `adaf/adaf_inference.py` | ADAF, STONE-modified | Main ADAF SLRM/tiling, OD/seg inference, vectorization | Active; correctness concerns | DEM/visualisation, models, `ADAFInput` | tiles, prediction files, GPKG, log | `adaf/stone_inference.py`, inference notebooks, ViT inference | Keep and test; fix aggregation by class/tile | High |
| `adaf/adaf_utils.py` | ADAF, unstaged local fix | Prediction helpers, VRT, logging, input object, raster clipping/tiling | Active; local fix not committed | rasters, model object, paths | TIF/VRT/TXT/logs | `inference/utils.py` | Keep; split responsibilities and test nodata/range handling | High |
| `adaf/adaf_vis.py` | ADAF, STONE-modified | Tiled SLRM production with RVT | Active | DEM/VRT, extents, resolution | normalized SLRM TIFs/VRT | `create_visualisations.py`, inference wrappers | Keep; externalize visualisation config | High |
| `adaf/adaf_widget.py` | ADAF | Notebook GUI and input collection | Active entry UI; work at import expected | user-selected paths/options | invokes `main_routine` | `ADAF_widget.ipynb` | Keep; isolate UI from processing | High |
| `adaf/create_visualisations.py` | ADAF, STONE-modified | Public wrapper to generate tiled SLRM | Active | DEM, tile size, save dir | path dictionary/VRT | `adaf_inference.run_visualisations` | Merge duplicate wrapper logic | High |
| `adaf/create_patches.py` | ADAF, STONE-modified | General image, 3-band segmentation-mask, and OD label creation | Active but assumptions conflict with e4MSTP | raster, class vectors, split GPKG | TIF patches/masks, label TXT, tile GPKG | `grid_patches.py`, `create_singleclass_dataset.py`, notebook copies | Keep/refactor; parameterize bands and splits | High |
| `adaf/create_singleclass_dataset.py` | STONE committed | Positive train/validation and full test grid; path creation; patch wrapper | Candidate active implementation | extent/arch/split GDFs, raster | tile GDF and patches | `grid_patches.py`, `create_patches.py` | Keep as candidate; correct docs and add split tests | High |
| `adaf/grid_patches.py` | STONE committed + unstaged rewrite | Alternative learning-grid builder with buffered split rules | Experimental/competing; hard-coded `__main__` | raster, label GPKGs, split GPKG | tile GDF/GPKG with paths | above; `learning_dataset_grid.ipynb` | Merge only after split-policy decision | High |
| `adaf/grid_tools.py` | ADAF | Raster grids, valid-data polygons, tile/VRT footprints | Active utility | raster/tile directory | GDF/GPKG/VRT-related metadata | inference grid creation | Keep; test CRS/nodata edge cases | High |
| `stone_dataset.py` | STONE committed | AiTLAS binary segmentation dataset; image-mask pairing and mask-band selection | Active candidate; config semantics conflict | image/mask dirs and AiTLAS config | tensors/dataloader state | `STONE-data_loader.ipynb`, `STONE_retrain_OLD2.ipynb` | Keep/refactor; enforce pairing and empty-mask policy | High |
| `evaluate.py` | STONE-only untracked, external-derived header | Centroid and pixel metrics | Duplicate/experimental | polygons or boolean masks | TP/FP/FN/TN, IoU | `adaf/evaluate.py` | Choose one licensed source; archive duplicate | High |
| `adaf/evaluate.py` | STONE-only untracked derivative | Same metrics plus MultiPolygon helper | Candidate but algorithm risky | polygons/masks | metric dict/tuple | top-level `evaluate.py` | Refactor with deterministic matcher and tests | High |
| `adaf/eval.py` | STONE-only untracked | File-oriented wrapper for centroid + pixel evaluation | Experimental, hard-coded defaults/main | prediction/GT/split GPKGs | metrics printed/returned | new-eval notebooks | Keep logic only after redesign | High |
| `adaf/stone_inference.py` | STONE-only untracked | Hard-coded segmentation-only fork of ADAF inference | Superseded/experimental | fixed Windows paths and custom checkpoint | GPKG/logs | `adaf_inference.py` (large copy) | Diff, port intentional changes, then archive | High |
| `vit_adaf_standalone_inference.py` | STONE-only untracked | Standalone 300 px ViT segmentation over georeferenced raster | Experimental reusable candidate | visualisation, `.pt`, parameters | detection GPKG | VIT notebooks | Keep in STONE module; validate preprocessing and seams | High |
| `inference/utils.py` | Legacy pre-ADAF/AiTLAS | Patch prediction helpers | Superseded, path-fragile | patch folder/model | prediction dirs/files | `adaf_utils.py` | Archive after confirmation | High |
| `_test_crete_patches.py` | STONE-only untracked | Hard-coded multiprocessing patch driver | Manual script, not a test | private paths | many patches | dataset notebooks | Convert to fixture-based integration test/example | High |
| `_test_inference.py` | STONE-only untracked | Hard-coded end-to-end inference driver | Dangerous import-time execution | private DEM/model/output paths | full inference outputs | ADAF notebook | Move behind `main` guard; convert to opt-in integration test | High |
| `run_adaf.bat` | ADAF | Launch Jupyter at `ADAF_main.ipynb` | Active but environment mismatch | Conda installation | notebook process | README | Fix after deciding environment name | High |

### 5.2 Notebooks

All paths below were inspected as JSON. “Saved error” means the notebook file contains an error output; it does not mean the audit executed the cell.

| File | Origin | Purpose | Status | Inputs | Outputs | Overlaps with | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|---|
| `ADAF_main.ipynb` | ADAF | Markdown launch hub | Active but spelling/encoding stale | linked notebooks/media | none | README | Keep and update links/text | High |
| `ADAF_widget.ipynb` | ADAF, output modified | Display ADAF GUI | Active; one executed cell | `adaf.adaf_widget` | embedded widget output | `ADAF_main` | Keep; clear/regenerate output only after verification | High |
| `ADAF_notebook.ipynb` | ADAF | Programmatic ADAF demo | Broken as saved: absolute path, missing `out_dir` | DEM, ADAF models | GPKG path | widget/inference script | Fix into config-driven smoke example | High |
| `ADAF_trainning_samples.ipynb` | ADAF, locally modified | General visualisation and patch tutorial | Broken/stale signature and split documentation | DEM, label GPKG, output dir | visualisation and learning patches | STONE dataset notebooks | Repair as ADAF tutorial or supersede explicitly | High |
| `train_and_evaluate_object_detection.ipynb` | ADAF | AiTLAS FasterRCNN train/evaluate | Template; unexecuted; heading wrongly says segmentation | Mac-specific data/weights | checkpoints/metrics | inference OD notebook | Keep as template after parameterization | High |
| `train_and_evaluate_semantic_segmentation.ipynb` | ADAF, heavily modified locally | AiTLAS HRNet train/evaluate | Local experiment with out-of-order execution | Windows data/weights | checkpoints/IoU output | STONE retrain notebooks | Extract experiment config; restore reusable template separately | High |
| `inference/make_predictions_object_detection.ipynb` | Legacy AiTLAS | Patch-level OD prediction | Old, machine-specific | model TAR and patch dir | bbox text | ADAF inference | Archive after confirming no unique code | High |
| `inference/make_predictions_segmentation.ipynb` | Legacy AiTLAS | Patch-level mask prediction | Old, machine-specific | model TAR and patch dir | mask TIFs | ADAF inference | Archive after confirmation | High |
| `inference/train_and_evaluate_segmentation.ipynb` | Legacy AiTLAS | HRNet-style training example | Unexecuted; Linux paths | Irish data/checkpoint | checkpoint/metrics | semantic training notebook | Archive/reference | High |
| `STONE_create_learning_dataset.ipynb` | STONE committed, output/text modified | Documented single-class grid -> SLRM -> patches workflow | Principal STONE dataset notebook; saved `NameError`; ~0.97 MB output | extents, labels, split, DEM | tiles GPKG, SLRM VRT, patches | module and e4MSTP version | Keep; parameterize and make restartable | High |
| `STONE_create_learning_dataset-e4MSTP.ipynb` | STONE-only | 300 px e4MSTP dataset variant | Experimental; saved multiband shape error | split/labels/e4MSTP VRT | tiles/patches | STONE dataset notebook | Keep experiment; fix multiband writer before reuse | High |
| `learning_dataset_grid.ipynb` | STONE-only | Develop/test alternate buffered grid logic | Exploratory; 33 executed cells, out of order, saved `AttributeError` | SLRM VRT, labels, split | GPKGs/patches | `grid_patches.py`, `Untitled` | Archive after extracting chosen tests/logic | High |
| `STONE_train_new_model.ipynb` | STONE committed | Train/evaluate custom `StoneDatasetSegmentation` HRNet | Executed experiment with large training output | BiH v5 data, ADAF TAR | checkpoints; saved class IoU 0.640 | `train-eval_BIH_v5_run4` | Preserve as experiment record; extract config/manifest | High |
| `STONE_retrain.ipynb` | STONE-only | Early AiTLAS retraining | Experimental; barrow uses band 1; interrupted training | hard-coded data/model | checkpoint/metrics | old/reusable training notebooks | Archive after documenting intent | High |
| `STONE_retrain_OLD.ipynb` | STONE-only | Earlier retraining attempt | Superseded; multiple data-path cells; saved error | hard-coded data/model | checkpoint | same | Archive, not delete until researcher confirms | High |
| `STONE_retrain_OLD2.ipynb` | STONE-only | Inline custom dataset development + training | Superseded exploratory; saved error | hard-coded data/model | checkpoint | `stone_dataset.py`, data-loader notebook | Archive after code comparison | High |
| `STONE-data_loader.ipynb` | STONE-only | Develop/compare custom and AiTLAS dataset classes | Exploratory, 42 cells, out-of-order | training dirs/masks | tensors/training/checkpoints | `stone_dataset.py`, OLD2 | Extract only missing tests; archive | High |
| `train-eval_BIH_v5_run4.ipynb` | STONE-only | Named HRNet experiment on 256 px BiH v5 | Valuable experiment record; out-of-order | external dataset/checkpoint | saved foreground IoU 0.688 | STONE train notebook | Preserve under final/experiment notebooks with manifest | High |
| `vision-transformers-for-segmentation.ipynb` | STONE-only, adapted external tutorial | Base custom ViT segmentation workflow | Exploratory/copy-derived | PNG pairs/CSV | weights/metrics/plots | two VIT variants | Retain provenance; consolidate | Medium |
| `VIT_e4mstp.ipynb` | STONE-only | e4MSTP ViT training/evaluation | Most developed but has 3 saved errors/interruption and 1.28 MB output | 300 px PNG dataset | `.pt` weights, logs, plots | base and sample-added | Preserve; extract module/config; do not treat prose metrics as verified | High |
| `VIT_e4mstp_sample_display_added.ipynb` | STONE-only | Near-copy with sample display | 90% cell-set overlap with `VIT_e4mstp`; no saved errors | same | same | VIT e4MSTP | Merge useful display cell then archive variant | High |
| `test_vit_inferenc.ipynb` | STONE-only | One-cell standalone inference example | Unexecuted smoke example | placeholder raster/model/output | GPKG | ViT inference module | Keep as tested example after fixtures/config | High |
| `new_eval_measure_template.ipynb` | STONE-only | Develop centroid/pixel evaluation | Exploratory; saved `KeyError`; 2.49 MB output | result/GT/split GPKGs | multiple metric trials and masks | eval module/variants | Replace with parameterized analysis; preserve output separately | High |
| `new_eval_measure adaf_bih_0-256px.ipynb` | STONE-only | Evaluate ADAF BiH 256 result | Result snapshot; 78% code-cell overlap with siblings | external GPKGs | TP 1466, FP 594, FN 451 | four siblings | Preserve result, regenerate from config | High |
| `new_eval_measure adaf_bih_0-512px.ipynb` | STONE-only | Evaluate ADAF BiH 512 result | Result snapshot | external GPKGs | TP 1474, FP 345, FN 443 | siblings | Same | High |
| `new_eval_measure adaf_retrained_2-256px.ipynb` | STONE-only | Evaluate retrained model | Result snapshot | external GPKGs | TP 1364, FP 594, FN 553 | siblings | Same | High |
| `new_eval_measure irish.ipynb` | STONE-only | Evaluate Irish model | Result snapshot; one unexecuted code cell | external GPKGs | TP 742, FP 149, FN 1175 | siblings | Same | High |
| `new_eval_measure retrained_1.ipynb` | STONE-only | Evaluate retrained model | Result snapshot | external GPKGs | TP 1594, FP 482, FN 323 | siblings | Same | High |
| `Object_Detection_Evaluation.ipynb` | Untracked, provenance unclear | Raster pixel/object error analysis with tolerance radius | Separate experimental method; Windows paths; title only “Bug fixes applied” | GT/prediction rasters | error rasters/PNG/statistics | eval family, but not integrated | Researcher provenance review; archive or refactor | Medium |
| `reference_data_refinment.ipynb` | STONE-only | Model-assisted search for missed reference labels | Useful exploratory workflow; includes optional destructive DB replace cell | PostGIS, supplementary vectors, model results | candidate GPKG/PostGIS table | evaluation results | Keep logic; move DB write behind explicit CLI/config | High |
| `torchgeo_001_multimodal_sample.ipynb` | STONE-only | Copy selected samples for multimodal work | Exploratory, filesystem-copy workflow | vector sample lists and external tree | copied subset | torchgeo dataset notebook | Rehome/archive after manifesting copied files | High |
| `torchgeo_002_datasets.ipynb` | STONE-only | Prototype `RasterDataset`/sampler and plots | Exploratory; 2.59 MB saved output | external SLRM tree | plots/samples | multimodal notebook | Extract only if multimodal work continues | High |
| `echoes_vectors.ipynb` | STONE-only | Combine many GeoJSON result files into GPKG | Small utility notebook | `echoes_paths_s2q.txt` and result tree | combined GPKG | reference refinement | Convert to script/config or archive | High |
| `echoes_s2patches.ipynb` | STONE-only | Inspect/plot ECHOES Sentinel-2 patch vector | Incomplete exploratory notebook | external GPKG | plot/display only | ECHOES workflow | Researcher review/archive | Medium |
| `Untitled.ipynb` | Ignored local experiment | VRT, extent-grid, split experiments | Unnamed; saved `KeyboardInterrupt`; no Markdown | private rasters/GPKG | VRT/grid objects | `grid_patches.py`, learning grid notebook | Rename and archive only after researcher identifies value | High |

### 5.3 Documentation, configuration, binaries, and generated material

| File/area | Origin/purpose | Status and recommendation |
|---|---|---|
| `README.md` | ADAF user installation/overview | Tracked, 15 upstream commits behind. Says create Conda env `adaf`; launcher activates `aitlas`. Bring upstream version and reconcile instructions. |
| `ADAF_manual_v1.1.pdf` | ADAF manual, 2.11 MB | Keep. Content comparison is unresolved due unavailable PDF tooling. |
| `LICENSE.txt` | Apache-2.0 repository license | Keep. Separately verify provenance/license obligations for copied evaluation and VIT tutorial code. |
| `.gitignore` | Mostly generic Python ignore file | Keep but revise intentionally: missing `.DS_Store`, `.vscode/`, `*.pth*`, outputs; ignores all `configs`, all `test*.py`, and `Untitled.ipynb`, which can hide useful source. |
| `.pylintrc` | Generic lint configuration | Keep if pylint remains selected; no automated lint/CI uses it. |
| `_wheels/*`, `installation/*` | Vendored GDAL/AiTLAS installers | Three large GDAL copies plus duplicate AiTLAS wheel. Upstream already removed `_wheels` and installation GDAL. Move to release/artifact storage only after offline-support decision. |
| `adaf/media/*` | GUI/docs images and STONE dataset diagram | Keep source media. `create_singleclass_dataset.png` documents STONE flow. |
| `.DS_Store`, `inference/.DS_Store` | Finder metadata | Tracked generated files; remove only after confirmation and add ignore rule. |
| `ADAF.lnk` | Windows shortcut | Machine-specific binary; inspect target manually, then keep as release aid or remove after confirmation. |
| `.vscode/settings.json` | Local editor environment-manager preference | Untracked and generic; either ignore or adopt shared settings intentionally. |
| `.idea/*` | Ignored IDE state | Local/generated; keep ignored. |
| `.ipynb_checkpoints/*`, `__pycache__/*` | Ignored generated state | Do not use as source unless recovering a missing edit; remove only after provenance snapshot. |
| `echoes_paths_s2q.txt` | 1,671 private absolute GeoJSON paths | Treat as local manifest with environment-specific/path-disclosure risk; replace with relative/generated manifest if workflow is retained. |

## 6. STONE workflow reconstruction

### 6.1 Shared geospatial preparation

```text
DEM / visualisation raster
  -> SLRM tiles + VRT (adaf_vis / create_visualisations)
  -> split-aware tile footprints (one of three builders)
  -> image clips + 3-band masks + optional OD labelTxt (create_patches)
  -> external train/validation/test directories
  -> HRNet or FasterRCNN training; alternatively PNG conversion -> custom ViT
  -> external checkpoint
  -> tiled inference
  -> GPKG detections
  -> post-filtering + centroid/pixel/object statistics
```

**Source spatial data -> visualisations.** `adaf/create_visualisations.py::run_visualisations` and `adaf/adaf_vis.py::tiled_processing` create SLRM. The SLRM radius is `ceil(10/resolution)` below 1 map unit/pixel, otherwise 10 cells; values are normalized from `[-0.5, 0.5]`, NaNs become 0, and a VRT is assembled. `adaf_inference.run_visualisations` builds 1024-pixel base tiles and buffers their geometries by 32 CRS units before RVT processing. An existing visualisation instead passes through `run_tiling()` and `clip_tile()`.

**Visualisations/labels -> patches.** `create_one_patch()` clips the image, rasterizes label polygons with `all_touched=True`, always writes a nominal three-band uint8 mask, and optionally writes OD label text. Labels retain `DFM` values (default 1); OD objects with no more than one-third of their original area in a patch are omitted by `prepare_labeltxt()`.

**Patches -> splits.** There are three competing paths:

1. `create_patches.py::create_patches_main`: positive tiles only, 512 px with 256 px stagger, exact validation/test containment; boundary-crossing tiles are discarded; other tiles are training.
2. `create_singleclass_dataset.py::build_singleclass_segmentation_grids`: 512 px/256 px overlap by default at a hard-coded/default 0.5 map units/pixel; training and validation are positive-only, test is full coverage; training area is geometry difference from validation/test.
3. Worktree `grid_patches.py::build_learning_dataset_grid`: configurable size/overlap and 10-unit inner/outer buffers; positive train/validation plus a shrunken full test area; ambiguous boundary behavior discussed below.

The split GeoPackage and actual generated `tiles.gpkg` are external. No repository manifest ties a particular experiment to one implementation or function version.

**Datasets -> training.** AiTLAS notebooks construct datasets from parallel image/mask or image/label directories. HRNet training uses `StoneDatasetSegmentation` or the AiTLAS `TiiLIDARDatasetSegmentation`; OD uses `TiiLIDARDatasetObjectDetection` and FasterRCNN. The ViT path first expects/creates PNG image-mask pairs and CSV lists, then trains a custom transformer.

**Checkpoints -> inference/evaluation.** ADAF uses model TAR/PTH paths outside Git. HRNet/FasterRCNN prediction helpers write masks or bbox text. ADAF converts these to GPKG. ViT standalone inference directly tiles a visualisation into non-overlapping 300 px windows, thresholds sigmoid output, polygonizes, dissolves, filters, and writes a GPKG.

**Detections/masks -> statistics.** Three paths exist: AiTLAS internal model metrics; centroid/pixel metrics in `evaluate.py`/`adaf/evaluate.py`/`adaf/eval.py`; and the separate raster/object analysis in `Object_Detection_Evaluation.ipynb`.

### 6.2 Missing/manual stages

- Creating or validating the spatial split polygons is manual and undocumented as a controlled artifact.
- Converting e4MSTP raster patches to the exact 3-channel PNG representation used by ViT is notebook-only and failed in the saved dataset notebook before later cells apparently used an external prepared dataset.
- Data transfer, model-weight acquisition, checkpoint naming, and result import to PostGIS are external/manual.
- No workflow records checksums, package lock, random seed, Git commit, command, or input-layer versions in a run manifest.
- No script connects evaluation result names to exact checkpoints and dataset manifests.

## 7. Notebook assessment

### 7.1 Execution state

The following notebooks contain saved errors or interrupted runs:

| Notebook | Saved issue |
|---|---|
| `STONE_create_learning_dataset.ipynb` cell 8 | `NameError: gpd is not defined` (evidence of out-of-order/older state; current import cell does import it) |
| `STONE_create_learning_dataset-e4MSTP.ipynb` cell 26 | `ValueError` for source shape inconsistent with three output indexes |
| `learning_dataset_grid.ipynb` cell 29 | `AttributeError: 'str' object has no attribute 'cx'` |
| `new_eval_measure_template.ipynb` cell 5 | `KeyError: 'area'`, yet later cells retain outputs from continued execution |
| `STONE_retrain.ipynb` cell 16 | `KeyboardInterrupt` during training |
| `STONE_retrain_OLD.ipynb` | saved error/interrupted state |
| `STONE_retrain_OLD2.ipynb` | saved error |
| `Untitled.ipynb` cell 24 | `KeyboardInterrupt` |
| `VIT_e4mstp.ipynb` cells 6, 9, 39 | missing Kaggle file, undefined `full_df`, then interrupted evaluation |

Execution counts are non-monotonic in `ADAF_trainning_samples`, both principal STONE dataset notebooks, all STONE retraining/data-loader notebooks, `learning_dataset_grid`, both current training experiment notebooks, the VIT family, `Object_Detection_Evaluation`, `torchgeo_001`, and `Untitled`. This confirms hidden-state risk; it does not by itself prove a scientific result is wrong.

### 7.2 Duplication by code-cell comparison

- The five named `new_eval_measure ...` notebooks share 7 of 9 distinct code cells (Jaccard 0.78; 88% of the smaller cell set). Only result paths and saved outputs materially vary.
- `VIT_e4mstp.ipynb` and `VIT_e4mstp_sample_display_added.ipynb` share 28 cells; all distinct cells in the smaller set are present in the larger comparison (Jaccard 0.90).
- The base ViT notebook shares 21 exact code cells with each e4MSTP variant.
- `STONE_retrain_OLD.ipynb` shares seven exact cells with the current semantic-training notebook.
- `STONE-data_loader.ipynb` and `STONE_retrain_OLD2.ipynb` contain long inline versions of logic now in `stone_dataset.py`.

### 7.3 Title/description mismatches

- `train_and_evaluate_object_detection.ipynb` begins “Example ... image segmentation” although it uses FasterRCNN and OD labels.
- `ADAF_trainning_samples.ipynb` says splitting is not covered by a Python script, but the current `create_patches_main()` requires a split GPKG; its call no longer matches that signature.
- `create_singleclass_dataset.py` documents a `(train_gdf, val_gdf, test_gdf)` tuple but returns one combined GeoDataFrame.
- The ViT notebooks retain tutorial prose about “building segmentation,” synthetic word OCR preparation, Kaggle/upvotes, and unverified prose metrics alongside STONE code.
- `Untitled.ipynb` has no Markdown and no stable purpose label.

### 7.4 Saved result evidence and limits

- `STONE_train_new_model.ipynb` saved sample counts `[4312, 1475, 1562]` and an evaluated IoU output with background `0.9975`, foreground `0.6401`, mean `0.8188`.
- `train-eval_BIH_v5_run4.ipynb` saved counts `[7048, 2730, 2797]` and IoUs background `0.9952`, foreground `0.6880`, mean `0.8416`.
- The five centroid notebooks save TP/FP/FN values listed in the inventory. They all use `iou_threshold=0` and the same test ground-truth count of 1,917 after spatial selection.
- `VIT_e4mstp.ipynb` has an interrupted validation evaluation. Markdown claiming IoU 0.57/Dice 0.70/pixel accuracy 0.89 cannot be linked to a completed saved output in the current notebook and should not be cited as the current e4MSTP result.

These are confirmed historical outputs, not independently validated results. Exact external data and checkpoints are missing.

## 8. Scripts and modules assessment

### Confirmed correctness/maintenance concerns

1. `ADAF_notebook.ipynb` does not set `out_dir`; `main_routine()` immediately does `Path(inp.out_dir)` (`adaf_inference.py`, lines 488-500).
2. `ADAF_trainning_samples.ipynb` cell 9 supplies three arguments to the four-argument `create_patches_main(input_raster, seg_masks_dict, split_gpkg, output_directory)`.
3. `_test_inference.py` calls a full inference run at module import because it has no `if __name__ == '__main__'` guard.
4. `adaf/eval.py` uses ambiguous `import evaluate`; behavior depends on launch directory and could resolve an unrelated installed package rather than the adjacent copy.
5. `create_segmentation_mask()` derives its array from the source image and repeats the band axis three times, but sets the output profile count to 3. This works for one-band input and fails for three/multiband input; the e4MSTP saved error is corroborating runtime evidence.
6. `StoneDatasetSegmentation.__getitem__()` repeats a one-band image to three channels but does not reduce 2, 4, or more bands. The comment says “Make sure it has 3 bands,” which the code does not guarantee.
7. `StoneDatasetSegmentation.load_dataset()` does not call `should_include_mask()`. Thus `keep_empty_patches=False` is ignored despite being set in training, validation, and test configs.
8. Image/mask pairing silently skips malformed/missing pairs and silently chooses one image on key collision. It raises only when the entire dataset is empty.
9. `clip_tile(..., out_nodata=0)` clips all raster values to `[0,1]` (`adaf_utils.py`, lines 497-503). This is appropriate only if every input visualisation already uses that range; it would binarize most of an 8-bit 0-255 visualisation.
10. `run_adaf.bat` activates `aitlas`, while README installation creates `adaf`.

### Likely interpretation

The code grew through notebook-driven experiments and was copied into modules when a path became useful. That explains why module code is generally clearer than notebook precursors but semantics were not consolidated. The best candidates for canonical code are the main ADAF modules, `create_singleclass_dataset.py` or a corrected merger with `grid_patches.py`, `stone_dataset.py`, and `vit_adaf_standalone_inference.py`; this choice still needs researcher confirmation.

## 9. Object-detection pipeline

### Reconstructed pipeline

1. Create SLRM patches using `create_visualisations`/`create_patches`.
2. Create `labelTxt` records with box corner coordinates, class, and DFM using `prepare_labeltxt()`; omit objects with <=33% of their area in a tile.
3. Load image/label directories with AiTLAS `TiiLIDARDatasetObjectDetection` (`train_and_evaluate_object_detection.ipynb`, cells 2, 5, 6).
4. Train/evaluate FasterRCNN, optionally starting from `OD_ringfort.tar` (cells 8-17).
5. At inference, tile DEM/visualisation, run one binary model per requested label, and write per-tile bbox text (`adaf_utils.make_predictions_on_patches_object_detection`).
6. `object_detection_vectors()` applies score `>0.5`, converts pixels to CRS boxes, unions overlaps per text file, optionally filters area, and writes `object_detection.gpkg`.

### Reliability assessment

- **Confirmed:** the tracked OD training notebook is an unexecuted template with another user's absolute macOS paths; there is no traceable STONE OD training result.
- **Confirmed:** OD inference does not globally deduplicate across files/overlapping tiles. Buffered inference tiles can therefore produce duplicate objects.
- **Confirmed:** `min_area` is documented as “max = 40 m2” but is implemented only as a lower threshold; the wording is wrong.
- **Confirmed:** labels are binary models with hard-coded class-to-weight filenames; model files are external.
- **Unresolved:** AiTLAS bbox augmentation/normalization, mAP definition, NMS settings, and checkpoint selection are hidden in the vendored wheel and were not runtime-inspected.
- **Unresolved:** `Object_Detection_Evaluation.ipynb` may be intended for this pipeline, but it consumes classification rasters, not ADAF bbox GPKGs, and no adapter is present.

## 10. Semantic-segmentation pipeline

### HRNet/AiTLAS path

1. Generate SLRM and three-band DFM-coded masks.
2. Use `StoneDatasetSegmentation` to select a class band/DFM quality and one-hot encode Background/Archaeology.
3. Train HRNet with learning rate `1e-4`, threshold `0.5`, IoU metric, and `FlipHVRandomRotate` for training only.
4. Load external TAR/PTH checkpoint; run AiTLAS mask prediction.
5. Crop eight pixels from every probability-map edge, threshold at 0.5, polygonize, dissolve, calculate area/roundness, filter, and write GPKG.

### ViT path

1. Use 300 px e4MSTP PNG image/mask pairs and CSV file lists.
2. `CustomDataset` converts images with `ToTensor`, divides mask by 255, and unfolds each 300 px image into 900 non-overlapping 10 px tokens.
3. Train a six-encoder, 256-embedding transformer with BCE+Dice, AdamW `1e-4`, AMP, batch size 16, and validation batch size 4 for 50 epochs.
4. Standalone inference reads up to three raster bands, heuristically scales integer data, tiles into non-overlapping 300 px windows, thresholds at 0.5, polygonizes/dissolves, and filters at area 40/roundness 0.5 by default.

### Reliability assessment

- **Confirmed:** HRNet's active current barrow config is band 0/DFM 1. Earlier `STONE_retrain.ipynb` uses band 1 and DFM 1,2. Results from those notebooks are not directly comparable without confirming mask schema.
- **Confirmed:** train/validation/test set enumeration in `stone_dataset.py` depends on unsorted `os.listdir`; training shuffle is unseeded. The ViT file collector also defaults to `sort=False` and training shuffle is unseeded.
- **Confirmed:** ViT training hard-fails unless CUDA is used, despite constructing an auto-selected device.
- **Confirmed:** ViT “best” checkpoints are considered only every tenth epoch, then both train-best and validation-best output paths are overwritten with the same final state when epoch 50 is reached.
- **Confirmed:** ViT `evaluate_model()` averages batch-level metrics rather than accumulating a global confusion matrix. This is equal to sample averaging only when batch sizes are equal; the current evaluation batch size is 1.
- **Confirmed:** cell 38 creates a test loader, but cell 39 replaces it with a validation loader before the interrupted evaluation. Saved display outputs reflect different hidden states.
- **Likely:** standalone ViT normalization is compatible with uint8 PNG `ToTensor`, but exact compatibility for e4MSTP GeoTIFF/VRT inputs is not demonstrated by a shared preprocessing test.
- **Likely:** butt-jointed ViT inference tiles can create boundary artifacts because there is no overlap/halo. Padding at raster edges is also treated as value 0.

## 11. Dataset and geospatial processing

| Topic | Finding | Status |
|---|---|---|
| Split construction | Three implementations use different containment, margin, positive/negative, and naming rules | Confirmed |
| Actual cross-split leakage | Data and tile GeoPackages absent; cannot compute intersections/duplicate prefixes | Unresolved |
| Buffered split rule | In `grid_patches`, a tile crossing an outer split boundary can miss `within(outer)` yet avoid the shrunken inner area and become training | Confirmed code behavior; actual occurrence unresolved |
| Split CRS | Worktree `grid_patches` reads and buffers `split_gdf` before reprojection in the initial filter; `_assign_splits` later reprojects a separate read | Confirmed; impact depends on CRS equality |
| Overlap | Common configs use 50% overlap: 512/256, 256/128, or 128/64; e4MSTP uses 300/0 | Confirmed |
| Duplicate grids | Multiple/overlapping validation or test polygons can emit duplicate tiles; tile IDs are not uniqueness-checked | Confirmed possibility; actual duplicates unresolved |
| Negative samples | `create_patches_main` is positive-only; newer builders include full test negatives but positive-only train/validation | Confirmed |
| Empty-mask handling | Dataset config says false, but custom loader does not filter empty masks | Confirmed |
| Nodata | Image nodata is converted to 0 and masks have no nodata value/valid-data mask | Confirmed |
| Raster range | `clip_tile(out_nodata=0)` clamps to `[0,1]` | Confirmed |
| Raster alignment | Raster windows derive from polygon bounds; there is no explicit integer-window/alignment assertion or output-size assertion | Confirmed absence; actual misalignment unresolved |
| Resolution | Single-class grid and pixel evaluation default/hard-code 0.5 units/pixel; no projected-unit validation | Confirmed |
| Mask rasterization | `all_touched=True`; later overlapping polygons overwrite earlier DFM values | Confirmed |
| Mask bands | Always nominally three classes/bands even in single-class workflow; >3 label layers are not validated | Confirmed |
| Tile IDs | Integer-truncated lower-left coordinates; uniqueness/collision not checked | Confirmed |
| Source provenance | No checksums, source dates, layer versions, or CRS/resolution manifest | Confirmed |

The split algorithms appear intended to avoid source-area leakage, but that intent is not enough to establish validity. The decisive P0 check is a geometry/ID audit on the exact generated dataset used for every reported model.

## 12. Evaluation methods and statistics

### 12.1 AiTLAS training metrics

HRNet notebooks request `metrics=["iou"]`; the saved outputs report a mean and per-class IoU. The mean includes background, which is near 0.995-0.998 and materially inflates the headline mean relative to archaeology IoU. Any report should expose foreground IoU separately. FasterRCNN requests `map`; the precise averaging/thresholds are unresolved without AiTLAS source/runtime validation.

### 12.2 Centroid metric

`compute_iou_metric_centroid()` treats a prediction as matched when its centroid is strictly contained in a ground-truth polygon and IoU meets the threshold. The result notebooks use threshold 0. It attempts one-to-one matching by list removal, but removal happens inside iteration and without a `break`.

Consequences:

- matching depends on input geometry order;
- overlapping/nested ground truths can be consumed unpredictably by one prediction;
- a centroid on a boundary fails because `contains`, not `covers`, is used;
- over-segmentation is usually penalized after the first match, but behavior with overlapping GT is not reliably one-to-one;
- under-segmentation and duplicate predictions need explicit test cases;
- invalid geometries are caught in `compute_iou()` by returning NaN, silently turning them into non-matches.

### 12.3 File-oriented evaluation wrapper

`adaf/eval.py::run_eval_metrics` applies fixed post-processing: area `<1500`, area `>=10`, roundness `>0.7`, exact split value `test`, raster resolution 0.5, and `all_touched=True`. It does not reproject GT/split layers to prediction CRS. Spatial joins select any object intersecting the test union but do not clip vector geometries before object-level matching. Pixel masks are rasterized per test rectangle and accumulated; overlapping test rectangles would double count pixels. It returns raw counts and leaves F1/export as TODOs.

Empty lists are mostly handled by the core centroid function, but the wrapper has no explicit contract/tests for empty prediction layers, empty GT, no test polygons, absent `area`/`roundness`, or invalid/mixed geometry types.

### 12.4 Alternative raster/object notebook

`Object_Detection_Evaluation.ipynb` computes pixel classification errors, tolerance-radius statistics, IoU-like measures, and per-connected-object accuracy from two rasters. It has useful unique logic but is disconnected from the vector evaluation and has unclear provenance, class semantics, radius units, and relationship to STONE results.

### 12.5 Traceability of reported results

The repository preserves result numbers but not the exact input GPKGs, checkpoints, environment, data manifest, or commit at run time. Some notebook paths identify result names (`adaf_bih_0-512px`, `retrained_1`, etc.), but this is insufficient for reproduction. Until reconstructed, rankings should be labeled historical/indicative.

## 13. Reproducibility assessment

| Requirement | Assessment | Evidence |
|---|---|---|
| Environment specification | **Not reproducible** | No `requirements`, lockfile, Conda YAML, package metadata, or CI; vendored AiTLAS has broad/unpinned dependencies and README only gives partial install commands |
| Data versioning | **Not reproducible** | All data are external absolute paths; no checksums/manifests |
| Model versioning | **Not reproducible** | Weights external; generic `checkpoint.pth.tar`; no hashes |
| Configuration capture | **Partial** | Values embedded in notebooks, frequently overwritten/out of order |
| Randomness | **Not reproducible** | No seed/deterministic settings in HRNet or ViT training; only a separate TorchGeo prototype seeds a generator |
| Preprocessing parity | **Unresolved/partial** | HRNet internals in AiTLAS; ViT has separate notebook and standalone implementations without equivalence test |
| Split integrity | **Unresolved** | No dataset; competing algorithms; no intersection/duplicate audit |
| Metrics | **Defined but unreliable without revision** | source exists, but greedy centroid order and hard-coded thresholds lack tests |
| Result provenance | **Insufficient** | saved outputs not tied to immutable data/model/config/code identifiers |
| Automated tests | **Absent** | no test runner config or fixtures; two hard-coded scripts are not tests |

## 14. Duplication, obsolete work, and conflicting implementations

### Consolidation candidates

| Area | Competing implementations | Decision needed |
|---|---|---|
| Dataset grids | `create_patches_main`, `create_singleclass_dataset`, `grid_patches`, `learning_dataset_grid`, `Untitled` | Authoritative split semantics and whether negatives belong in train/validation |
| Dataset class | `stone_dataset.py`, inline OLD2 class, two classes in data-loader notebook, AiTLAS class | Mask schema, empty-mask behavior, filename contract |
| Semantic training | ADAF semantic notebook, three STONE retrain notebooks, STONE new model, BIH run4 | Template vs immutable experiment records |
| ViT | base notebook, two e4MSTP variants, standalone module | Source provenance and canonical architecture/preprocessing |
| Evaluation | top-level/`adaf` duplicate `evaluate.py`, `adaf/eval.py`, six notebooks, object raster notebook | Metric specification and output schema |
| Inference | `adaf_inference.py`, `stone_inference.py`, legacy inference notebooks/utils, ViT module | Keep ADAF core, isolate STONE adapters, port only intentional fork changes |

No file should be deleted solely because it looks unused. “OLD,” checkpoint, and ignored files may contain provenance. Archive them only after an experiment manifest records what they contributed and a researcher confirms the canonical successor.

## 15. Risks and correctness concerns

| Priority | Finding | Evidence | Status/impact |
|---|---|---|---|
| P0 | Current results lack immutable data/split/model lineage | external paths and absent manifests | Confirmed; prevents independent reproduction |
| P0 | Potential split leakage at buffered boundaries and duplicates | `_assign_splits_to_grid`, no uniqueness/intersection checks | Code risk confirmed; actual dataset impact unresolved |
| P0 | Multiband mask generation fails | `create_segmentation_mask`; e4MSTP cell 26 error | Confirmed |
| P0 | Class-band inconsistency | barrow band 0 vs band 1 across notebooks | Confirmed; can train on wrong class |
| P0 | Multi-class semantic dissolve loses label separation | ungrouped `gdf.dissolve()` | Confirmed code behavior when classes overlap |
| P0 | OD duplicates across buffered tiles | only per-file union | Confirmed code behavior; frequency unresolved |
| P0 | Greedy centroid metric is order-dependent | list removal during iteration | Confirmed |
| P0 | Reference refinement includes `to_postgis(... if_exists="replace")` | notebook cell 25 | Confirmed data-loss risk if executed; not executed in audit |
| P1 | Empty-mask flag ignored | custom loader comments/code | Confirmed |
| P1 | Nodata and valid-area handling conflated with background | clip/mask code | Confirmed; statistical effect unresolved |
| P1 | Evaluation CRS/resolution assumptions unchecked | `adaf/eval.py` | Confirmed |
| P1 | No seeds/environment lock | repository-wide search | Confirmed |
| P1 | ViT final checkpoint overwrites “best” files | `train_model` cell 30 | Confirmed |
| P1 | Current ADAF demo/training entry notebooks are broken | missing output dir / signature mismatch | Confirmed |
| P1 | Saved metrics can be misread because background dominates mean | saved HRNet outputs | Confirmed |
| P1 | Database connection configuration is stored in `reference_data_refinment.ipynb` cell 2 | credentials are read from environment; connection metadata remains notebook-specific | Confirmed location/type; no secret values reproduced here |
| P2 | Copied code diverges silently | duplicate metrics/inference/dataset/VIT | Confirmed |
| P2 | Repository-root/CWD dependency | ambiguous import, notebook paths, no package config | Confirmed |
| P2 | Generated output bloats notebooks and can expose paths/coordinates | saved outputs and absolute-path manifest | Confirmed |
| P2 | Large installers are duplicated and upstream already removed them | Git size/history | Confirmed |
| P3 | Encoding corruption and filename spelling variants | worktree Markdown, `trainning`, `refinment`, `inferenc`, `crete`, `splti` | Confirmed; maintenance/searchability impact |

## 16. Proposed target repository structure

Do not implement this structure until the P0/P1 decisions are made.

```text
src/
  adaf/                         # reusable upstream-derived ADAF library
    inference.py
    visualisation.py
    raster.py
    grid.py
    ui.py
  stone/                        # STONE-specific reusable library
    data/
      grids.py
      patches.py
      datasets.py
      validation.py
    models/
      hrnet.py
      vit.py
    inference/
      hrnet.py
      vit.py
      vectorize.py
    evaluation/
      matching.py
      pixel.py
      reporting.py
workflows/
  dataset_preparation/
  training/
  inference/
  evaluation/
configs/                        # tracked resolved experiment configs
  datasets/
  experiments/
notebooks/
  exploratory/
  final_analysis/
tests/
  unit/
  integration/
  fixtures/
docs/
  audits/
  workflows/
  decisions/
outputs/                        # ignored; reports exported deliberately
data/                           # ignored/local or DVC/object storage
models/                         # ignored/local or artifact registry
```

Before using `configs/`, remove the current broad `configs` ignore rule.

### Current-to-target mapping

| Current files | Proposed destination |
|---|---|
| `adaf/adaf_inference.py`, `adaf_utils.py`, `adaf_vis.py`, `grid_tools.py`, `adaf_widget.py` | `src/adaf/` split by responsibility, retaining upstream provenance |
| `adaf/create_visualisations.py` | thin `workflows/dataset_preparation/` entry point calling `src/adaf/visualisation.py` |
| chosen parts of `create_patches.py`, `create_singleclass_dataset.py`, `grid_patches.py` | one `src/stone/data/` implementation plus config-driven workflow |
| `stone_dataset.py` and validated inline improvements | `src/stone/data/datasets.py` |
| `vit_adaf_standalone_inference.py` and canonical notebook architecture | `src/stone/models/vit.py` + `src/stone/inference/vit.py` |
| `adaf_inference` STONE-specific vectorization fixes | shared or `src/stone/inference/vectorize.py`, explicitly tested by class |
| `evaluate.py`, `adaf/evaluate.py`, `adaf/eval.py` | one `src/stone/evaluation/` package with deterministic matching and CLI |
| ADAF entry notebooks | `notebooks/final_analysis/adaf/` or root launch stubs if required for users |
| STONE dataset/training/result notebooks | stable templates in `workflows/`; immutable run narratives in `notebooks/final_analysis/` |
| OLD, data-loader, grid, TorchGeo, ECHOES, and Untitled notebooks | `notebooks/exploratory/<topic>/` or an archive tag/branch after researcher review |
| `_test_*.py`, one-cell VIT test | `tests/integration/` with tiny synthetic fixtures and explicit slow markers |
| README/manual/audit | `docs/` plus concise root README |
| wheel files | release assets/artifact store; preserve installation instructions/checksums |
| data, weights, predictions, TensorBoard logs, notebook-generated CSV/GPKG/TIF | ignored `data/`, `models/`, `outputs/` or managed external storage |

## 17. Prioritized action plan

| Priority/action | Affected files | Reason | Expected benefit | Risk of change | Prerequisite decisions | Verification criterion |
|---|---|---|---|---|---|---|
| **P0 - snapshot provenance** | all 6 modified + 32 untracked files; external datasets/models/results | Cleanup/branch updates can orphan the only record | Protects experiments and enables review | Snapshot may include sensitive paths/large output | Approved storage location and confidentiality level | Immutable archive/branch has hashes and status manifest; restore test succeeds |
| **P0 - create run manifests** | STONE train, BIH run4, VIT, new-eval notebooks | Results lack exact inputs/code/config | Makes each result auditable | May reveal private paths; use logical IDs | Which runs are publication-relevant | Each retained result names Git SHA, data/split/model hashes, environment, seed, config |
| **P0 - validate exact split datasets** | both grid modules, `create_patches.py`, generated `tiles.gpkg` | Leakage/duplicates cannot be excluded | Protects scientific validity | Reclassification may invalidate results | Authoritative dataset version and intended boundary/negative policy | Automated report shows unique IDs and zero cross-split geometry/overlap above agreed tolerance |
| **P0 - decide and test class/mask schema** | `create_patches.py`, `stone_dataset.py`, retrain notebooks, e4MSTP | Band conflict and multiband failure | Prevents training on wrong labels | Existing checkpoints may become incompatible | Mapping of classes, DFM values, image channels | Synthetic 1/3/N-band tests pass; visual sample audit signed off; checkpoint input shape matches |
| **P0 - repair inference aggregation** | `adaf_inference.py`, `stone_inference.py`, ViT vectorization | Duplicate OD and cross-class semantic dissolve | Correct object counts/classes | Changes historical results | Desired per-class merge/NMS and seam policy | Synthetic overlapping-tile/multiclass fixtures preserve labels and produce agreed one-to-one objects |
| **P0 - specify/reimplement matching** | both evaluate modules, `adaf/eval.py`, evaluation notebooks | Current metric order-dependent | Comparable, defensible statistics | Rankings may change | Centroid vs IoU rule, one-to-one optimizer, thresholds, empty cases | Permutation-invariant tests cover over/under-segmentation, duplicates, empty GT/pred, boundaries |
| **P0 - guard database writes** | `reference_data_refinment.ipynb` | `if_exists="replace"` can destroy a table | Protects research database | Adds workflow friction | Approved schema/table/write policy | Default run is read-only/dry-run; destructive write requires explicit confirmation and backup |
| **P1 - lock environment** | README, launcher, new env/lock files, wheel policy | No reproducible runtime | Repeatable execution | Legacy GDAL/CUDA constraints | Supported OS/Python/CUDA matrix and AiTLAS distribution | Fresh environment runs import/smoke tests from documented commands |
| **P1 - choose canonical dataset builder** | three implementations and notebook precursors | Semantics conflict | One documented workflow | Unique useful behavior may be lost | Split/negative/overlap policy | Golden fixture produces expected tiles/paths with one implementation |
| **P1 - separate config from notebooks** | all active workflows | Hard-coded paths/overwritten state | Portable, reviewable runs | Migration mistakes | Config schema and path abstraction | Notebooks contain no machine-specific required values; resolved config is saved per run |
| **P1 - fix loader contracts** | `stone_dataset.py` | silent pair drops, unordered pairs, ignored flag | Stable sample identity and honest options | Dataset length may change | Empty-patch policy and collision rule | Loader emits manifest/counts, errors on collisions/missing pairs, honors flag |
| **P1 - establish metric reports** | eval package/notebooks | Raw counts and background-heavy means are easy to misread | Clear foreground/object/pixel results | Historical numbers may differ | Primary and secondary metrics | Report includes definitions, thresholds, CIs/sample counts, foreground and background separately |
| **P1 - repair entry examples** | ADAF demo/training notebooks, `run_adaf.bat`, README | Current documented routes fail/mismatch | Restores user path | Could diverge from upstream | Whether root remains ADAF app or STONE research repo | Static smoke test validates arguments; launcher uses documented environment |
| **P2 - consolidate copied code** | inference, dataset, eval, VIT families | Divergence/maintenance risk | Single tested source of truth | Loss of provenance | P0 manifests complete | Notebook imports modules; duplicate-cell report falls below agreed threshold |
| **P2 - reorganize with history** | all files | Mixed library/experiments/generated assets | Discoverability | Moves break links/imports | Target structure approval | Link/import/path tests pass; mapping document covers every moved item |
| **P2 - create real tests/CI** | `_test_*`, core modules | No automated safety net | Safer refactoring | Geospatial fixtures/dependencies take effort | Environment lock and canonical semantics | Unit suite runs read-only on tiny fixtures; slow tests are opt-in |
| **P2 - artifact/ignore policy** | wheels, outputs, checkpoints, IDE/cache/path list | Git bloat and accidental disclosure | Cleaner repository | Offline installation could break | Artifact host and retention | Clone + documented artifact fetch reproduces environment; sensitive/local files stay ignored |
| **P3 - notebook hygiene** | all active notebooks | saved errors/output/encoding obscure intent | Smaller, reviewable notebooks | Removing output can erase evidence | Results exported first | Clean copy executes top-to-bottom in controlled environment; output reports stored separately |
| **P3 - spelling/docs cleanup** | filenames, docstrings, README/manual | Searchability and credibility | Easier navigation | Renames break links/history | Structure decision | Link checker and import tests pass |

## 18. Questions requiring researcher input

### Three decisions required before cleanup

1. **Which dataset/split definition is authoritative?** Choose the exact source raster/visualisation, label snapshot, split GeoPackage, tile size/overlap, margin rule, whether train/validation include negative patches, and whether test polygons should be fully sampled. Is BiH v5, v5-256, v7-128, `stone-singleclass_v0`, or `vit_300px_e4mstp` the reference dataset?
2. **Which experiments/results must remain scientifically citable?** Identify the canonical HRNet run/checkpoint, whether ViT and object detection are active research lines, and which of the five centroid-evaluation result sets belong in final analysis. This determines what must be manifested versus archived.
3. **What is the target relationship to upstream ADAF?** Should this repository stay an ADAF fork that periodically merges `main`, become a STONE package depending on ADAF, or split reusable STONE code into a separate repository? Also decide whether offline vendored wheels must remain supported.

Additional unresolved questions:

- Is `DFM` a data-quality selector, class attribute, or both, and should values `1,2` be included in current STONE training?
- Are barrow/enclosure/ringfort always bands 0/1/2, and should AO mean their union?
- Is e4MSTP expected to be a 3-band model input, 9-band input, or a rendered RGB product?
- What geometric buffer/tolerance is scientifically intended between spatial splits?
- Which matching rule should be primary: centroid containment, IoU assignment, pixel IoU, or tolerance-radius object detection?
- Should model-assisted missed labels require agreement from multiple models, and what manual adjudication is required before adding them to reference data?
- Do the copied IIT evaluation and ViT tutorial portions have provenance/license notes that must be retained?
- Does the ADAF PDF manual still describe the supported environment and model paths accurately?

## 19. Appendix of commands and evidence used

Representative read-only commands (paths/large output abbreviated in this report):

```powershell
git status --short --branch
git status --porcelain=v2 --branch
git remote -v
git branch -a -vv
git rev-parse HEAD
git log --all --graph --decorate --date=short --oneline
git ls-remote --symref https://github.com/EarthObservation/adaf.git HEAD refs/heads/main
git merge-base HEAD origin/main
git rev-list --left-right --count HEAD...origin/main
git diff --name-status 925493e..HEAD
git diff --name-status 815df52..origin/main
git diff --name-status
git diff --stat
git ls-files
git ls-files --others --exclude-standard
git ls-files --others --ignored --exclude-standard
git ls-tree -r -l HEAD
git count-objects -vH
git diff --check
git fsck --no-reflogs --connectivity-only
rg --files -uu
rg -n "^(class |def |if __name__)" -g "*.py"
rg -n -i "manual_seed|random.seed|deterministic|threshold|roundness|min_area" -g "*.py" -g "*.ipynb"
```

Notebook JSON was parsed with PowerShell `ConvertFrom-Json` to enumerate cells, sources, imports, definitions, paths, execution counts, outputs, and errors. Exact code-cell hashes were compared pairwise while ignoring output and metadata. No source file was imported or executed.

Other evidence:

- `git ls-remote` verified default branch and tip against the expected public URL without adding/changing a remote.
- Wheel ZIP `METADATA` shows AiTLAS `0.0.1`, Python `>=3.6`, a mixture of broad minimum versions and a few exact pins; it is not a full environment lock.
- The root excluding `.git` measured 97,102,998 bytes; Git objects measured 91.60 MiB packed and 10.70 MiB loose.
- PDF tooling and Python were unavailable; no package was installed in accordance with the audit constraints.

### Preservation statement

During the audit, no pre-existing repository file was modified, moved, renamed, formatted, deleted, staged, committed, fetched into this repository, or executed as a notebook. The only repository file created is this report: `docs/audits/STONE_REPOSITORY_AUDIT.md`. The upstream check used a read-only remote query and did not alter persistent remotes.
