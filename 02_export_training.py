import os
import numpy as np

from graph_utils import (
    build_graph_from_fc, generate_candidates, label_candidates_from_full,
    mine_hard_negatives, split_train_val_indices, write_json, msg,
)
from feature_registry import DEFAULT_REGISTRY

WORK_GDB = r"C:\Users\Imran\Documents\ArcGIS\Projects\MyProject.gdb"
FULL_FC = os.path.join(WORK_GDB, 'prep_full_road')
TRAIN_FC = os.path.join(WORK_GDB, 'prep_train_road')
FULL_NODE_FC = os.path.join(WORK_GDB, 'prep_full_nodes')
TRAIN_NODE_FC = os.path.join(WORK_GDB, 'prep_train_nodes')
FULL_CAND_FC = os.path.join(WORK_GDB, 'prep_full_candidates')
TRAIN_CAND_FC = os.path.join(WORK_GDB, 'prep_train_candidates')

EXPORT_DIR = r"C:\GIS\Project\GeoGNN_Publishable_v2\export"
EXPORT_NPZ = os.path.join(EXPORT_DIR, 'training_export.npz')
EXPORT_META = os.path.join(EXPORT_DIR, 'training_export.json')
ROUND_DIGITS = 2
MIN_GAP = 1.0
MAX_GAP = 120.0
REQUIRE_DEAD_END = True
MAX_ANGLE = 60.0
DIFFERENT_COMPONENTS_ONLY = False
HARD_NEG_RATIO = 3
VAL_RATIO = 0.20
RANDOM_SEED = 42


def main():
    full_g = build_graph_from_fc(FULL_FC, DEFAULT_REGISTRY, ROUND_DIGITS, node_fc=FULL_NODE_FC)
    train_g = build_graph_from_fc(TRAIN_FC, DEFAULT_REGISTRY, ROUND_DIGITS, node_fc=TRAIN_NODE_FC)

    cand_pairs, cand_feat, cand_meta = generate_candidates(
        train_g,
        DEFAULT_REGISTRY,
        MIN_GAP,
        MAX_GAP,
        REQUIRE_DEAD_END,
        MAX_ANGLE,
        different_components_only=DIFFERENT_COMPONENTS_ONLY,
    )
    if len(cand_pairs) == 0:
        raise RuntimeError('No training candidates were generated. Relax export gap/angle settings and rerun 00_prepare_data.py and 02_export_training.py.')

    y_raw = label_candidates_from_full(cand_pairs, full_g, train_g)
    msg(f'Raw training candidates generated: {len(cand_pairs)}')
    msg(f'Raw positive missing edges: {int(y_raw.sum())}')

    cand_pairs, cand_feat, y, keep_idx = mine_hard_negatives(
        cand_pairs, cand_feat, cand_meta, y_raw, neg_ratio=HARD_NEG_RATIO, random_seed=RANDOM_SEED
    )
    cand_meta = [cand_meta[int(i)] for i in keep_idx.tolist()]

    tr_idx, va_idx = split_train_val_indices(len(y), val_ratio=VAL_RATIO, random_seed=RANDOM_SEED)
    if len(tr_idx) == 0:
        raise RuntimeError('Train/validation split produced zero training examples.')

    os.makedirs(EXPORT_DIR, exist_ok=True)
    np.savez(
        EXPORT_NPZ,
        x=train_g['x'],
        edge_index=train_g['edge_index'],
        edge_attr=train_g['edge_attr'],
        cand_pairs_train=cand_pairs[tr_idx],
        cand_feat_train=cand_feat[tr_idx],
        y_train=y[tr_idx],
        cand_pairs_val=cand_pairs[va_idx],
        cand_feat_val=cand_feat[va_idx],
        y_val=y[va_idx],
        cand_pairs_all=cand_pairs,
        cand_feat_all=cand_feat,
        y_all=y,
    )

    meta = {
        'round_digits': ROUND_DIGITS,
        'min_gap': MIN_GAP,
        'max_gap': MAX_GAP,
        'require_dead_end': REQUIRE_DEAD_END,
        'max_angle': MAX_ANGLE,
        'different_components_only': DIFFERENT_COMPONENTS_ONLY,
        'hard_neg_ratio': HARD_NEG_RATIO,
        'val_ratio': VAL_RATIO,
        'random_seed': RANDOM_SEED,
        'node_feature_names': train_g['feature_names']['node'],
        'road_feature_names': train_g['feature_names']['road'],
        'candidate_feature_names': train_g['feature_names']['candidate'],
        'prepared_full_fc': FULL_FC,
        'prepared_train_fc': TRAIN_FC,
        'prepared_full_nodes_fc': FULL_NODE_FC,
        'prepared_train_nodes_fc': TRAIN_NODE_FC,
        'prepared_full_candidates_fc': FULL_CAND_FC,
        'prepared_train_candidates_fc': TRAIN_CAND_FC,
        'raw_candidate_count': int(len(y_raw)),
        'raw_positive_count': int(y_raw.sum()),
        'hardmined_candidate_count': int(len(y)),
        'hardmined_positive_count': int(y.sum()),
        'train_count': int(len(tr_idx)),
        'val_count': int(len(va_idx)),
        'workflow_note': 'Use 00_prepare_data.py first so full/train/unseen are synchronized before graph export.',
    }
    write_json(EXPORT_META, meta)

    msg(f'Prepared training road FC: {TRAIN_FC}')
    msg(f'Prepared full road FC: {FULL_FC}')
    msg(f'Observed graph nodes: {train_g["x"].shape[0]}')
    msg(f'Observed undirected edges: {len(train_g["road_records"])}')
    msg(f'Full undirected edges: {len(full_g["road_records"])}')
    msg(f'Hard-negative mined candidate count: {len(y)}')
    msg(f'Positives after mining: {int(y.sum())}')
    msg(f'Train examples: {len(tr_idx)} | Val examples: {len(va_idx)}')
    msg(f'Saved NPZ: {EXPORT_NPZ}')
    msg(f'Saved metadata: {EXPORT_META}')


if __name__ == '__main__':
    main()
