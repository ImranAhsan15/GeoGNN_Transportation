# GeoGNN Publishable v2 — matched to your current file system

This package keeps your current 00→05 workflow and file names, but upgrades the modeling logic:

- `00_prepare_data.py`: prepare full, train-incomplete, and unseen-incomplete FCs consistently
- `01_make_supervised_examples.py`: optional synthetic missing-example generation
- `02_export_training.py`: export graph + supervised candidate labels with **hard-negative mining**
- `03_train_model.py`: train model with **saved scaler** and **validation-based threshold**
- `04_prepare_unseen.py`: package unseen graph from prepared FCs
- `05_predict_unseen.py`: load model + scaler, score candidates, **prune best connections**, and save FCs

## Why this matches your system
It preserves:
- the same script names
- the same prepared FC names (`prep_full_*`, `prep_train_*`, `prep_unseen_*`)
- the same `feature_registry.py`, `graph_utils.py`, `models.py` pattern
- the same ArcGIS inspection stage with RF_*, NF_*, CF_* fields

## Dynamic fields
Edit `feature_registry.py`.

### Built-in toggles
Turn built-in features on/off by changing `True/False` in:
- `road_features`
- `node_features`
- `candidate_features`

### Extra dynamic GIS fields
You can also add your own fields without breaking the workflow:
- `extra_road_fields = ['WIDTH', 'TYPE_CODE']`
- `extra_node_fields = ['manual_score', 'dem_z']`
- `extra_candidate_fields = ['cnn_support', 'mask_prob']`

How they are used:
- extra road fields are read directly from `prep_*_road`
- extra node fields are read from `prep_*_nodes` as `NF_<name>`
- extra candidate fields are read from `prep_*_candidates` as `CF_<name>`

After editing the registry, rerun `00_prepare_data.py`.

## Important modeling upgrades
- hard negatives instead of mostly easy negatives
- standardization fit on training only, reused at prediction
- threshold selected from validation instead of fixed 0.5/0.8
- best-per-endpoint pruning to reduce messy overconnection

## Run order
1. `00_prepare_data.py`
2. optional `01_make_supervised_examples.py`
3. `02_export_training.py`
4. `03_train_model.py`
5. `04_prepare_unseen.py`
6. `05_predict_unseen.py`

## Outputs
- export/training NPZ + JSON
- model checkpoint
- scaler NPZ
- scored candidate FC
- predicted missing-edge FC
- restored network FC
