# Reliability-Aware Early Correctness Recognition for Exercise Feedback under Partial 3D Skeleton Observation

Reproducibility code for the MAPR 2026 camera-ready paper by Dinh-Huu-Phuc Nguyen, Thi-Kieu Tran, Ngoc-Thuy Nguyen, and Duc-Nhat-Quang Nguyen (University of Sciences, Hue University).

The study evaluates correctness recognition from 25%, 50%, 75%, and 100% of a processed 3D skeleton repetition. Its main result is **75.46 ± 1.49% correctness macro-F1 and 75.56 ± 1.48% accuracy at the 75% prefix** for PA-MTTCN. The evaluation uses three seeds and fixed subject partitions. Feedback readiness and risk are measured offline.

## Rebuild Tables I–VIII

Run these commands from the repository root with Python 3.10. The bundled CSVs contain predictions from the camera-ready experiments. This path requires NumPy and pandas.

```sh
python -m pip install -r requirements.txt
python -m src.eval_camera_ready.build_camera_ready_results
```

The script recomputes the metrics and writes CSVs to `runs/paper-tables/tables/`. Reference CSVs from the camera-ready run are in [results/paper_tables](results/paper_tables/). Rates in the machine-readable columns are fractions between 0 and 1.

| Paper table | Generated CSV prefix | Content |
| --- | --- | --- |
| I | `table1_` | PA-MTTCN action macro-F1, correctness macro-F1, and correctness accuracy |
| II | `table2_` | Correctness macro-F1 for all four comparison models |
| III | `table3_` | Zero-filled versus mask-aware input |
| IV | `table4_` | Readiness sensitivity to the threshold |
| V | `table5_` | Per-exercise test support and readiness at τ = 0.80 |
| VI | `table6_` | Pooled post-hoc test confidence sweep at the 75% prefix |
| VII | `table7_` | Validation-selected confidence gate by seed and prefix |
| VIII | `table8_` | Test calibration, including per-seed results |

### Table VIII calibration

The reported Table VIII values are reproduced by fitting a temperature separately for each seed and prefix on validation predictions. Section IV and the Table VIII caption describe a single temperature fitted to pooled validation predictions. The following command evaluates that described procedure in a separate output directory. It fits **T = 1.74** from 1,416 validation predictions in the supplied files. At 75%, test ECE after calibration is 0.1446 with pooled fitting and 0.1375 in the camera-ready table.

```sh
python -m src.eval_camera_ready.build_camera_ready_results --calibration pooled --output-dir runs/pooled-calibration
```

Tables VI and VII use the original sigmoid probabilities in both runs. Table VII maximizes validation positive-feedback coverage subject to FPR ≤ 10% for each seed and prefix, with ties going to the lower threshold. When the grid has no feasible threshold, the script chooses minimum validation FPR, then maximum coverage, then the lower threshold.

## Retrain from the processed skeletons

Retraining starts from the original **1,057 processed NumPy arrays** of shape `(64, 26, 3)`. The arrays are external inputs. Place them under `data/processed/rehab24/samples/` or pass their directory to `--samples-dir`. The packaged [metadata](data/processed/rehab24/metadata.csv) defines the original input order, labels, subjects, and prefixes. The preparation command verifies every array against [samples.sha256](data/processed/rehab24/samples.sha256).

The official REHAB24-6 download contains the upstream recordings, skeletons, and annotations. The original conversion to the fixed-length arrays was not retained, so the official download alone is insufficient for exact retraining. See the dataset record for terms of use and attribution. [REHAB24-6 on Zenodo](https://zenodo.org/records/13305826)

```sh
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m src.eval_camera_ready.prepare_camera_ready_run
python -m src.train_camera_ready.train_val_selected_models --device cpu
python -m src.eval_camera_ready.build_camera_ready_results --predictions-dir runs/camera-ready/predictions --output-dir runs/retrained-tables
```

For arrays stored elsewhere, run `python -m src.eval_camera_ready.prepare_camera_ready_run --samples-dir "/path/to/samples"` in place of the preparation command above.

The default run trains PA-MTTCN, TCN-C, LSTM-C, LSTM-MT, and mask-aware PA-MTTCN with seeds 42, 2024, and 3407. Subjects 1–7 provide 804 training repetitions, subject 8 provides 118 validation repetitions, and subjects 9–10 provide 135 test repetitions. Training uses all four prefixes, 30 epochs, AdamW, batch size 32, learning rate 0.001, and weight decay 0.0001. Checkpoints minimize the validation objective. Validation and test predictions are exported from the selected checkpoint. The original run used Python 3.10.9 and PyTorch 2.12.1+cpu on Windows. Retraining on a different software or hardware configuration can change predictions.

## Citation

Use [CITATION.cff](CITATION.cff) to cite the paper. For REHAB24-6, cite the dataset and its accompanying publication as specified in the [dataset record](https://zenodo.org/records/13305826).
