"""Build the unseen graph package from prepared unseen FCs.

Run 00_prepare_data.py first with MODE='unseen_side_only' or MODE='all'.
This script does not recopy raw roads; it uses the synchronized prepared outputs.
"""
import os
import numpy as np

from graph_utils import build_graph_from_fc, generate_candidates, msg, write_json
from feature_registry import DEFAULT_REGISTRY

WORK_GDB = r"C:\Users\Imran\Documents\ArcGIS\Projects\MyProject.gdb"
PREP_UNSEEN_ROAD = os.path.join(WORK_GDB, 'prep_unseen_road')
PREP_UNSEEN_NODES = os.path.join(WORK_GDB, 'prep_unseen_nodes')
PREP_UNSEEN_CANDS = os.path.join(WORK_GDB, 'prep_unseen_candidates')

UNSEEN_DIR = r"C:\GIS\Project\GeoGNN_Publishable_v2\unseen"
UNSEEN_NPZ = os.path.join(UNSEEN_DIR, 'unseen_graph_package.npz')
UNSEEN_META = os.path.join(UNSEEN_DIR, 'unseen_graph_package.json')
ROUND_DIGITS = 2
MIN_GAP = 1.0
MAX_GAP = 120.0
REQUIRE_DEAD_END = True
MAX_ANGLE = 60.0
DIFFERENT_COMPONENTS_ONLY = False


def main():
    g = build_graph_from_fc(PREP_UNSEEN_ROAD, DEFAULT_REGISTRY, ROUND_DIGITS, node_fc=PREP_UNSEEN_NODES)
    cand_pairs, cand_feat, _ = generate_candidates(
        g,
        DEFAULT_REGISTRY,
        MIN_GAP,
        MAX_GAP,
        REQUIRE_DEAD_END,
        MAX_ANGLE,
        different_components_only=DIFFERENT_COMPONENTS_ONLY,
    )
    if len(cand_pairs) == 0:
        raise RuntimeError('No unseen candidates were generated from prep_unseen_road. Relax gap/angle filters or inspect prep_unseen_nodes/candidates in ArcGIS Pro.')

    os.makedirs(UNSEEN_DIR, exist_ok=True)
    np.savez(
        UNSEEN_NPZ,
        x=g['x'],
        edge_index=g['edge_index'],
        edge_attr=g['edge_attr'],
        cand_pairs=cand_pairs,
        cand_feat=cand_feat,
    )
    meta = {
        'prepared_unseen_fc': PREP_UNSEEN_ROAD,
        'prepared_unseen_nodes_fc': PREP_UNSEEN_NODES,
        'prepared_unseen_candidates_fc': PREP_UNSEEN_CANDS,
        'round_digits': ROUND_DIGITS,
        'min_gap': MIN_GAP,
        'max_gap': MAX_GAP,
        'require_dead_end': REQUIRE_DEAD_END,
        'max_angle': MAX_ANGLE,
        'different_components_only': DIFFERENT_COMPONENTS_ONLY,
        'node_feature_names': g['feature_names']['node'],
        'road_feature_names': g['feature_names']['road'],
        'candidate_feature_names': g['feature_names']['candidate'],
    }
    write_json(UNSEEN_META, meta)
    msg(f'Prepared unseen road FC: {PREP_UNSEEN_ROAD}')
    msg(f'Prepared unseen nodes FC: {PREP_UNSEEN_NODES}')
    msg(f'Prepared unseen candidates FC: {PREP_UNSEEN_CANDS}')
    msg(f'Unseen graph package saved: {UNSEEN_NPZ}')
    msg(f'Unseen metadata saved: {UNSEEN_META}')
    msg(f'Unseen candidates generated for prediction: {len(cand_pairs)}')


if __name__ == '__main__':
    main()
