# Maize phenology transferability

Data and scientific code supporting **Predictive skill and environmental transferability of maize phenology models**. Repository: [SmartAG-Team/maize-phenology-transferability](https://github.com/SmartAG-Team/maize-phenology-transferability). The [v1.0.0 Release](https://github.com/SmartAG-Team/maize-phenology-transferability/releases/tag/v1.0.0) contains the complete model-ready study dataset, fitted models, native predictions, frozen memberships and weather-comparison inputs. All supporting data and code are available through this repository and its Release assets.

The comparisons use source-qualified German maize event dates. They assess linear and nonlinear thermal-time references, residual RF/MLP, a causal thermal Transformer, source-support blending, local historical updating, annual-temperature transfer and DWD versus ERA5-Land forcing. Native BBCH 87 remains distinct from harvest or generic physiological maturity. Supplied-weather retrospective skill is not operational forecast skill.

Version **v1.0.1** adds a head-balanced violin distribution to Figure 5, paired from zero to five historical local years. Full model inputs and fitted results retain the checked **v1.0.0** data assets.

## Quick reproduction

Python 3.12 and the pinned dependencies are recorded in `requirements.txt`. From the repository root:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_release.py
.venv/bin/python replay_weather_example.py
.venv/bin/python replay_local_example.py
.venv/bin/python scripts/render_figures.py
.venv/bin/python scripts/reconstruct_metrics.py
.venv/bin/python -m pytest tests/test_annual_temperature_transfer.py tests/test_expanded_phenology_results.py -q
```

These commands use the compact evidence in Git. They regenerate all 10 main figures and Figure S1, reconstruct 924 grouped numerical comparisons and conditional intervals, and replay saved-model predictions. The examples contain 24 native units for all six weather models and three sites for nonlinear-T local updating across 0/1/3/5 historical-year budgets. They verify computation and do not constitute additional independent accuracy estimates.

## Complete study data and fitted cases

```sh
.venv/bin/python scripts/restore_full_data.py --download
```

Alternatively, download the three `.tar.gz` assets from the Release page into one directory and use `scripts/restore_full_data.py --assets /path/to/assets`. `full_data_assets.json` records their sizes, SHA-256 hashes and download URLs. The script verifies every asset before extraction and refuses unsafe paths or changes to existing local files.

| Release asset | Contents after restoration |
| --- | --- |
| `study_model_ready_data.tar.gz` | `data/study_dataset/`: all 207,325 exact-date labels, 47,008 native units at 2,232 German site groups, native daily Tmin/Tmax/daylength arrays, valid lengths, pre-2011 chronological masks, context variables, all five frozen source roles, weather lineage and site/fold metadata. |
| `annual_temperature_transfer_results.tar.gz` | `results/annual_temperature_transfer_20261024/`: all annual-zone case fits and keyed predictions, annual climate inputs, structural/spatial/local analyses, DWD station archives, common DWD/ERA5 arrays and paired weather refits. |
| `year_matched_exploration_results.tar.gz` | `results/phenology_exploration_20261023/year_matched/`: complete source-budget benchmark, frozen source memberships, temperature-response follow-up fits, neural fits, local updating and auxiliary numerical analyses. |

The dataset export preserves all numerical arrays exactly and verifies an exact CSV round trip. Native weather padding is retained; `lengths` and `train_lengths` specify usable daily arrays and the chronological source boundary. Provider definitions, day conventions and attribution remain distinct. Complete model-ready inputs are supplied; unused observations from other countries and unrelated acquired collections are outside this study. Original numerical/binary fitted artifacts are retained byte for byte. Per-file manifests record path sanitization in text provenance and exclusions of manuscript, administrative and review material.

## Inspect inputs and refit

After restoring the full assets, `research.phenology_exploration.data.load()` reads the portable dataset; `research.annual_temperature_transfer.design.dataset()` applies the frozen annual-temperature zones. Original scientific fitting, teacher prediction, selection, inference, aggregation and resampling modules are under `research/`. The legacy `maize_clock/` identifier is retained for software compatibility.

The following commands validate a frozen case's inputs, source/target separation, inner block roles and membership hash without fitting:

```sh
.venv/bin/python scripts/refit_case.py --cohort annual --prepare-only
.venv/bin/python scripts/refit_case.py --cohort dwd --prepare-only
.venv/bin/python scripts/refit_case.py --cohort era5 --prepare-only
```

Remove `--prepare-only` to refit all six models for that case. Use `--case NAME` for another frozen case listed in the relevant `cases.json`. New fits go to `outputs/refits/`, preserving released fits. Training is computationally expensive; hardware/library differences can affect fitted weights. Numerical reproduction of the reported comparisons uses the frozen released predictions. Original fitting drivers retain strict code/input receipt guards; changed portable paths or code hashes do not silently qualify as the original calculation.

## Files and interpretation

- `plot_data/` contains exact figure workbooks and captions; generated figures and numerical reconstructions go to `outputs/`.
- `evidence/` contains frozen qualification, memberships, site/head error tables and summary comparisons. The large climate-transfer table is split into byte-exact chunks and automatically reassembled locally.
- `SOURCE_PROVENANCE.json`, `release_manifest.json` and the Release manifests document the source snapshot, released files and portable adaptations.
- `DATA_SOURCES.md` records provider attribution and source-specific reuse terms. Saved model files originate from this snapshot; use the checked manifests and a trusted repository copy.

Invalid event dates incur 366 days loss; weighting follows observations within site-year, years per site, sites per head and equal native heads. Target-block intervals condition on fitted models and are exploratory. Weather resampling additionally joins shared stations. Annual-temperature strata and observed transfer contrasts do not identify causal warming or cooling effects. Original event definitions, cohort boundaries and information budgets remain explicit.

Source data retain their own licences; no blanket project-wide software licence is inferred. No access credentials are included or required for offline reproduction.

Optional Earth Engine reacquisition requires the user’s own authorized credentials and project; set `PHENOLOGY_EE_PROJECT` to that project. The original acquisition project is recorded as provenance, and all required offline inputs are already supplied.

To reproduce the revised local-history figure alone:

```sh
.venv/bin/python scripts/render_figures.py --figure fig5
.venv/bin/python scripts/verify_figure5.py
```

Figure 5 retains the accuracy curve and adds all 5,323 matched site–endpoint gains per model, with equal endpoint weights, weighted medians and the original mean/block intervals. Its signed asinh axis retains every tail and expands the centre; density estimation uses untransformed days. Exact pairs, density curves, summaries and intervals are in `plot_data/fig5.xlsx`.

## Reproduction boundary

Version **v1.0.2** clarifies the boundary between offline numerical reproduction and source reacquisition. All reported model comparisons and figures can be reconstructed offline from the supplied qualified inputs, saved fits and keyed predictions. The reference climatology is supplied as frozen Earth Engine aggregates, calendar-completeness receipts and site assignments; the complete 1981–2010 daily ERA5-Land reference-grid series is not locally archived. Recomputing that source aggregation requires authorized access to the specified provider collection and a fresh output directory, preserving the original receipt guards. Source products retain their own versions and day conventions. Full model retraining and live weather reacquisition are separate operations from replaying frozen results.

## Workflow figure

Version **v1.0.3** improves Figure 1 with shared inputs and calibration controls, six consistently coloured model formulations, four parallel comparison cards and a common evaluation step. Conceptual node/edge coordinates, text placement and encodings are in `plot_data/fig1.xlsx`; they represent study design, not measured performance values.

```sh
.venv/bin/python scripts/render_figures.py --figure fig1
.venv/bin/python scripts/verify_workflow.py
```

## Figure title policy

Figures have no standalone titles. Descriptive titles belong in captions. Axis labels, legends, panel letters and labels inside workflow nodes retain their explanatory roles. The renderer rejects figure and axes titles; the workflow verifier rejects standalone text outside diagram nodes.

## Weather-effect distributions

Version **v1.0.5** displays the paired weather effects in Figure 10 as 18 violin distributions. Three source draws are averaged before weighting sites equally within each of seven equally weighted native endpoints. The original paired means and station/block confidence intervals remain unchanged; white diamonds mark descriptive medians. Gaussian density is computed in raw days; the signed-asinh display retains complete observed tails.

```sh
.venv/bin/python scripts/render_figures.py --figure fig10
.venv/bin/python scripts/verify_weather_violins.py
```
