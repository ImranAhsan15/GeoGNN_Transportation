"""Prepare road, node, and candidate FCs for training and/or unseen workflows.

Modes:
- training_side_only  (full + train)
- unseen_side_only    (unseen only)
- all                 (full + train + unseen)

This stage is for GIS inspection and manual feature editing before export/train/predict.
If you add extra fields in feature_registry.py, rerun this file so RF_*, NF_*, CF_* fields are recreated.
"""
import os
import arcpy

from feature_registry import DEFAULT_REGISTRY
from graph_utils import (
    msg, prepare_fc, materialize_road_features, save_node_fc, save_candidate_fc,
)

# =========================
# USER CONFIG
# =========================
WORK_GDB = r"C:\Users\Imran\Documents\ArcGIS\Projects\MyProject.gdb"
FULL_ROAD_FC = os.path.join(WORK_GDB, 'Road_Full')
TRAIN_INCOMPLETE_FC = os.path.join(WORK_GDB, 'Road_Missing_Train')
UNSEEN_INCOMPLETE_FC = os.path.join(WORK_GDB, 'Road_Missing_Unseen')


MODE = 'all'  # 'training_side_only' | 'unseen_side_only' | 'all'
PREPARE_ROADS = True
PREPARE_NODES = True
PREPARE_CANDIDATES = True

MIN_GAP_DISTANCE = 1.0
MAX_GAP_DISTANCE = 120.0
REQUIRE_ONE_DEAD_END = True
MAX_TERMINAL_ANGLE_DIFF = 60.0
DIFFERENT_COMPONENTS_ONLY = False
ROUND_DIGITS = 2

OUT_FULL_ROAD = os.path.join(WORK_GDB, 'prep_full_road')
OUT_FULL_NODES = os.path.join(WORK_GDB, 'prep_full_nodes')
OUT_FULL_CANDS = os.path.join(WORK_GDB, 'prep_full_candidates')

OUT_TRAIN_ROAD = os.path.join(WORK_GDB, 'prep_train_road')
OUT_TRAIN_NODES = os.path.join(WORK_GDB, 'prep_train_nodes')
OUT_TRAIN_CANDS = os.path.join(WORK_GDB, 'prep_train_candidates')

OUT_UNSEEN_ROAD = os.path.join(WORK_GDB, 'prep_unseen_road')
OUT_UNSEEN_NODES = os.path.join(WORK_GDB, 'prep_unseen_nodes')
OUT_UNSEEN_CANDS = os.path.join(WORK_GDB, 'prep_unseen_candidates')
# =========================


def ensure_gdb(gdb_path):
    folder = os.path.dirname(gdb_path)
    name = os.path.basename(gdb_path)
    if not arcpy.Exists(gdb_path):
        arcpy.management.CreateFileGDB(folder, name)


def prepare_one(label, src_fc, out_road, out_nodes, out_cands):
    msg(f'Preparing {label}: {src_fc}')
    prep_fc = prepare_fc(src_fc, out_road)
    graph = materialize_road_features(prep_fc, prep_fc, registry=DEFAULT_REGISTRY, round_digits=ROUND_DIGITS)

    if PREPARE_NODES:
        save_node_fc(graph, out_nodes, registry=DEFAULT_REGISTRY)
        msg(f'Saved nodes: {out_nodes}')

    if PREPARE_CANDIDATES:
        cands, _, _ = save_candidate_fc(
            graph, out_cands, registry=DEFAULT_REGISTRY,
            min_gap=MIN_GAP_DISTANCE,
            max_gap=MAX_GAP_DISTANCE,
            require_dead_end=REQUIRE_ONE_DEAD_END,
            max_angle=MAX_TERMINAL_ANGLE_DIFF,
            different_components_only=DIFFERENT_COMPONENTS_ONLY,
        )
        msg(f'Saved candidates: {out_cands} | count={len(cands)}')

    msg(f'Saved road: {out_road}')


def main():
    arcpy.env.overwriteOutput = True
    ensure_gdb(WORK_GDB)

    do_full = MODE in ('training_side_only', 'all')
    do_train = MODE in ('training_side_only', 'all')
    do_unseen = MODE in ('unseen_side_only', 'all')

    if do_full and PREPARE_ROADS:
        prepare_one('full/reference road', FULL_ROAD_FC, OUT_FULL_ROAD, OUT_FULL_NODES, OUT_FULL_CANDS)
    if do_train and PREPARE_ROADS:
        prepare_one('training incomplete road', TRAIN_INCOMPLETE_FC, OUT_TRAIN_ROAD, OUT_TRAIN_NODES, OUT_TRAIN_CANDS)
    if do_unseen and PREPARE_ROADS:
        prepare_one('unseen incomplete road', UNSEEN_INCOMPLETE_FC, OUT_UNSEEN_ROAD, OUT_UNSEEN_NODES, OUT_UNSEEN_CANDS)

    msg('00_prepare_data completed.')
    msg('Inspect RF_* on roads, NF_* on nodes, and CF_* on candidates in ArcGIS Pro.')
    msg('Dynamic fields are controlled in feature_registry.py.')
    if DEFAULT_REGISTRY.extra_node_fields:
        msg(f'Extra node fields enabled: {DEFAULT_REGISTRY.extra_node_fields}')
    if DEFAULT_REGISTRY.extra_road_fields:
        msg(f'Extra road fields enabled: {DEFAULT_REGISTRY.extra_road_fields}')
    if DEFAULT_REGISTRY.extra_candidate_fields:
        msg(f'Extra candidate fields enabled: {DEFAULT_REGISTRY.extra_candidate_fields}')


if __name__ == '__main__':
    main()
