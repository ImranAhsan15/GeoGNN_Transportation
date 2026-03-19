import json
import os

import numpy as np
import torch

try:
    import arcpy
except Exception:
    arcpy = None

from feature_registry import DEFAULT_REGISTRY
from graph_utils import (
    build_graph_from_fc, load_prepared_candidate_fc, prepared_nodes_summary,
    apply_standard_scaler, rank_and_prune_pairs, msg,
)
from models import CandidateScorer

WORK_GDB = r"C:\Users\Imran\Documents\ArcGIS\Projects\MyProject.gdb"
PREP_UNSEEN_ROAD = os.path.join(WORK_GDB, 'prep_unseen_road')
PREP_UNSEEN_NODES = os.path.join(WORK_GDB, 'prep_unseen_nodes')
PREP_UNSEEN_CANDS = os.path.join(WORK_GDB, 'prep_unseen_candidates')

EXPORT_META = r"C:\GIS\Project\GeoGNN_Publishable_v2\export\training_export.json"
MODEL_PATH = r"C:\GIS\Project\GeoGNN_Publishable_v2\model\geognn_publishable_v2.pt"
SCALER_PATH = r"C:\GIS\Project\GeoGNN_Publishable_v2\model\feature_scaler.npz"

OUT_SCORED_CAND_FC = os.path.join(WORK_GDB, 'pred_scored_candidates')
OUT_PRED_FC = os.path.join(WORK_GDB, 'pred_missing_edges')
OUT_RESTORED_FC = os.path.join(WORK_GDB, 'restored_network')
ROUND_DIGITS = 2
FALLBACK_THRESH = 0.85


def save_lines_from_pairs(base_fc, out_fc, graph, pairs, probs, keep_mask):
    sr = arcpy.Describe(base_fc).spatialReference
    ws, name = os.path.dirname(out_fc), os.path.basename(out_fc)
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.management.CreateFeatureclass(ws, name, 'POLYLINE', spatial_reference=sr)
    for fn, ftype in [
        ('CAND_ID', 'LONG'), ('U_NODE', 'LONG'), ('V_NODE', 'LONG'),
        ('CONF', 'DOUBLE'), ('KEEP', 'SHORT')
    ]:
        arcpy.management.AddField(out_fc, fn, ftype)
    with arcpy.da.InsertCursor(out_fc, ['SHAPE@', 'CAND_ID', 'U_NODE', 'V_NODE', 'CONF', 'KEEP']) as ic:
        for i, ((u, v), pr, kp) in enumerate(zip(pairs, probs, keep_mask)):
            p0 = graph['idx_to_node'][int(u)]
            p1 = graph['idx_to_node'][int(v)]
            arr = arcpy.Array([arcpy.Point(*p0), arcpy.Point(*p1)])
            ic.insertRow([arcpy.Polyline(arr, sr), i, int(u), int(v), float(pr), int(kp)])


def main():
    with open(EXPORT_META, 'r', encoding='utf-8') as f:
        train_meta = json.load(f)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    dims = ckpt['dims']
    chosen_threshold = float(ckpt.get('threshold', FALLBACK_THRESH))
    scaler = np.load(SCALER_PATH)
    mu = scaler['mean'].astype(np.float32)
    sigma = scaler['std'].astype(np.float32)

    if not arcpy.Exists(PREP_UNSEEN_ROAD):
        raise RuntimeError('Prepared unseen road FC not found. Run 00_prepare_data.py first.')
    if not arcpy.Exists(PREP_UNSEEN_NODES):
        raise RuntimeError('Prepared unseen node FC not found. Run 00_prepare_data.py with unseen preparation enabled.')
    if not arcpy.Exists(PREP_UNSEEN_CANDS):
        raise RuntimeError('Prepared unseen candidate FC not found. Run 00_prepare_data.py with unseen preparation enabled.')

    node_info = prepared_nodes_summary(PREP_UNSEEN_NODES)
    cand_pairs, cand_feat = load_prepared_candidate_fc(PREP_UNSEEN_CANDS, DEFAULT_REGISTRY)
    if len(cand_pairs) == 0:
        raise RuntimeError('Prepared unseen candidate FC contains zero candidates. Relax candidate settings in 00_prepare_data.py and rerun it.')

    graph = build_graph_from_fc(PREP_UNSEEN_ROAD, DEFAULT_REGISTRY, ROUND_DIGITS, node_fc=PREP_UNSEEN_NODES)
    x = graph['x'].astype(np.float32)
    edge_attr = graph['edge_attr'].astype(np.float32)
    if cand_feat.shape[1] != dims['cand']:
        raise RuntimeError(f'Candidate feature mismatch. Prediction has {cand_feat.shape[1]} fields but model expects {dims["cand"]}. Re-run 00_prepare_data.py, 02_export_training.py, and 03_train_model.py with the same feature registry.')

    cand_feat = apply_standard_scaler(cand_feat.astype(np.float32), mu, sigma)

    x_t = torch.tensor(x, dtype=torch.float32)
    edge_index_t = torch.tensor(graph['edge_index'], dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    cand_pairs_t = torch.tensor(cand_pairs, dtype=torch.long)
    cand_feat_t = torch.tensor(cand_feat, dtype=torch.float32)

    model = CandidateScorer(dims['node'], dims['edge'], dims['cand'], hidden=dims['hidden'])
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    with torch.no_grad():
        logits = model(x_t, edge_index_t, edge_attr_t, cand_pairs_t, cand_feat_t)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # reconstruct candidate metadata directly from FC field values for pruning
    cand_names = train_meta['candidate_feature_names']
    cand_meta = []
    present_names = [n for n in cand_names if any(f.name == f'CF_{n[:24]}' for f in arcpy.ListFields(PREP_UNSEEN_CANDS))]
    field_names = [f'CF_{n[:24]}' for n in present_names]
    with arcpy.da.SearchCursor(PREP_UNSEEN_CANDS, field_names) as cur:
        for row in cur:
            meta = {present_names[i]: float(row[i] or 0.0) for i in range(len(present_names))}
            cand_meta.append(meta)

    keep = rank_and_prune_pairs(cand_pairs, probs, cand_meta, chosen_threshold)

    msg(f'Prepared unseen road used for prediction: {PREP_UNSEEN_ROAD}')
    msg(f'Prepared unseen nodes used for prediction: {PREP_UNSEEN_NODES} | node_count={node_info["node_count"]}')
    msg(f'Prepared unseen candidate FC used for scoring: {PREP_UNSEEN_CANDS}')
    msg(f'Scored candidates: {len(probs)}')
    msg(f'Chosen model threshold: {chosen_threshold:.6f}')
    msg(f'Predicted positives after pruning: {int(keep.sum())}')
    msg(f'Probability range: min={float(np.min(probs)):.6f}, max={float(np.max(probs)):.6f}, mean={float(np.mean(probs)):.6f}')

    save_lines_from_pairs(PREP_UNSEEN_ROAD, OUT_SCORED_CAND_FC, graph, cand_pairs, probs, keep)
    keep_idx = np.where(keep == 1)[0]
    save_lines_from_pairs(
        PREP_UNSEEN_ROAD,
        OUT_PRED_FC,
        graph,
        cand_pairs[keep_idx],
        probs[keep_idx],
        np.ones(len(keep_idx), dtype=np.int32),
    )
    if arcpy.Exists(OUT_RESTORED_FC):
        arcpy.management.Delete(OUT_RESTORED_FC)
    arcpy.management.Merge([PREP_UNSEEN_ROAD, OUT_PRED_FC], OUT_RESTORED_FC)

    msg(f'Saved scored candidate FC: {OUT_SCORED_CAND_FC}')
    msg(f'Saved predicted FC: {OUT_PRED_FC}')
    msg(f'Saved restored FC: {OUT_RESTORED_FC}')


if __name__ == '__main__':
    main()
