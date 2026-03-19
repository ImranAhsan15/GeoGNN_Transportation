import json
import math
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np

try:
    import arcpy
except Exception:
    arcpy = None

from feature_registry import DEFAULT_REGISTRY, FeatureRegistry


# -------------------------------
# General helpers
# -------------------------------
def msg(s):
    if arcpy:
        try:
            arcpy.AddMessage(str(s))
            return
        except Exception:
            pass
    print(str(s))


def safe_exists(path):
    if arcpy:
        return arcpy.Exists(path)
    return os.path.exists(path)


def ensure_dir_for_file(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def write_json(path, obj):
    ensure_dir_for_file(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def field_exists(fc, name):
    return any(f.name.lower() == name.lower() for f in arcpy.ListFields(fc))


def add_field_safe(fc, name, ftype='DOUBLE', length=255):
    if not field_exists(fc, name):
        if ftype.upper() == 'TEXT':
            arcpy.management.AddField(fc, name, ftype, field_length=length)
        else:
            arcpy.management.AddField(fc, name, ftype)


def node_key(pt, digits=2):
    return (round(float(pt[0]), digits), round(float(pt[1]), digits))


def line_endpoints(geom):
    sp = geom.firstPoint
    ep = geom.lastPoint
    return (float(sp.X), float(sp.Y)), (float(ep.X), float(ep.Y))


def line_length(geom):
    try:
        return float(geom.length)
    except Exception:
        return 0.0


def bearing_deg(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return float(math.degrees(math.atan2(dy, dx)) % 180.0)


def angle_diff_180(a, b):
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def make_polyline(p0, p1, spatial_reference):
    arr = arcpy.Array([arcpy.Point(p0[0], p0[1]), arcpy.Point(p1[0], p1[1])])
    return arcpy.Polyline(arr, spatial_reference)


# -------------------------------
# FC readers / GIS preparation
# -------------------------------
def fc_to_segments(fc, round_digits=2, extra_fields=None):
    extra_fields = extra_fields or []
    rows = []
    fields = ['OID@', 'SHAPE@'] + [f for f in extra_fields if field_exists(fc, f)]
    with arcpy.da.SearchCursor(fc, fields) as cur:
        for rec in cur:
            oid, geom = rec[0], rec[1]
            if geom is None:
                continue
            try:
                if geom.partCount < 1:
                    continue
            except Exception:
                continue
            p0, p1 = line_endpoints(geom)
            attrs = {}
            for i, fname in enumerate(fields[2:], start=2):
                attrs[fname] = rec[i]
            rows.append({
                'oid': int(oid),
                'p0': p0,
                'p1': p1,
                'u': node_key(p0, round_digits),
                'v': node_key(p1, round_digits),
                'length': line_length(geom),
                'bearing': bearing_deg(p0, p1),
                'geom': geom,
                'input_attrs': attrs,
            })
    return rows


def prepare_fc(in_fc, out_fc, split_output=None):
    if safe_exists(out_fc):
        arcpy.management.Delete(out_fc)
    arcpy.management.RepairGeometry(in_fc, 'DELETE_NULL')
    arcpy.management.CopyFeatures(in_fc, out_fc)
    if split_output:
        if safe_exists(split_output):
            arcpy.management.Delete(split_output)
        arcpy.management.CopyFeatures(out_fc, split_output)
        return split_output
    return out_fc


# -------------------------------
# Graph building + dynamic node injection
# -------------------------------
def _node_override_map(node_fc, registry: FeatureRegistry):
    """Read manual NF_* fields from prepared node FC and index by NODE_KEY."""
    if not node_fc or (arcpy and not arcpy.Exists(node_fc)):
        return {}

    extra_names = list(registry.extra_node_fields)
    builtins = []
    # Allow manual overrides for disabled or external versions of existing names too.
    for n in registry.enabled_names('node_features'):
        if n not in ('x', 'y'):
            builtins.append(n)

    read_fields = ['NODE_KEY']
    out_names = []
    for n in builtins + extra_names:
        fn = f'NF_{n[:24]}'
        if field_exists(node_fc, fn):
            read_fields.append(fn)
            out_names.append(n)

    if len(read_fields) == 1:
        return {}

    out = {}
    with arcpy.da.SearchCursor(node_fc, read_fields) as cur:
        for row in cur:
            key_raw = row[0]
            vals = {}
            for i, nm in enumerate(out_names, start=1):
                vals[nm] = float(row[i] or 0.0)
            out[str(key_raw)] = vals
    return out


def build_graph_from_fc(fc, registry: FeatureRegistry = DEFAULT_REGISTRY, round_digits=2, node_fc=None):
    segs = fc_to_segments(fc, round_digits=round_digits, extra_fields=registry.extra_road_fields)
    node_to_idx = {}
    idx_to_node = []
    edges = []
    incident = defaultdict(list)
    for s in segs:
        for nk in [s['u'], s['v']]:
            if nk not in node_to_idx:
                node_to_idx[nk] = len(idx_to_node)
                idx_to_node.append(nk)
        u = node_to_idx[s['u']]
        v = node_to_idx[s['v']]
        edges.append((u, v, s))
        incident[u].append(s)
        incident[v].append(s)

    adj = defaultdict(list)
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)

    comp = {}
    cid = 0
    for n in range(len(idx_to_node)):
        if n in comp:
            continue
        q = deque([n])
        comp[n] = cid
        while q:
            a = q.popleft()
            for b in adj[a]:
                if b not in comp:
                    comp[b] = cid
                    q.append(b)
        cid += 1
    comp_sizes = defaultdict(int)
    for _, c in comp.items():
        comp_sizes[c] += 1

    node_overrides = _node_override_map(node_fc, registry)
    road_names = registry.all_road_names()
    node_names = registry.all_node_names()
    cand_names = registry.all_candidate_names()

    x = []
    node_attr_maps = []
    for i, nk in enumerate(idx_to_node):
        x0, y0 = nk
        deg = len(incident[i])
        lengths = [r['length'] for r in incident[i]] or [0.0]
        bearings = [r['bearing'] for r in incident[i]] or [0.0]
        neigh = adj.get(i, [])
        neigh_deg = [len(incident[j]) for j in neigh] or [0]
        local_deadend_ratio = float(sum(d <= 1 for d in neigh_deg) / max(len(neigh_deg), 1))
        local_junction_ratio = float(sum(d >= 3 for d in neigh_deg) / max(len(neigh_deg), 1))
        values = {
            'x': x0,
            'y': y0,
            'degree': float(deg),
            'dead_end': 1.0 if deg <= 1 else 0.0,
            'junction': 1.0 if deg >= 3 else 0.0,
            'component_size': float(comp_sizes[comp.get(i, -1)]),
            'incident_len_mean': float(np.mean(lengths)),
            'incident_len_max': float(np.max(lengths)),
            'incident_len_std': float(np.std(lengths)),
            'dominant_angle': float(np.mean(bearings)),
            'local_deadend_ratio': local_deadend_ratio,
            'local_junction_ratio': local_junction_ratio,
            'nearest_node_dist': 0.0,
            'elevation': 0.0,
        }
        key_str = str(nk)
        if key_str in node_overrides:
            values.update(node_overrides[key_str])
        for ef in registry.extra_node_fields:
            values.setdefault(ef, 0.0)

        feats = [float(values.get(n, 0.0)) for n in node_names]
        x.append(feats)
        node_attr_maps.append(values)

    edge_index = []
    edge_attr = []
    road_records = []
    for u, v, s in edges:
        p0, p1 = s['p0'], s['p1']
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        L = max(s['length'], 1e-6)
        vals = {
            'length': s['length'],
            'bearing': s['bearing'],
            'dx': dx,
            'dy': dy,
            'abs_dx': abs(dx),
            'abs_dy': abs(dy),
            'norm_dx': dx / L,
            'norm_dy': dy / L,
            'mid_x': (p0[0] + p1[0]) / 2.0,
            'mid_y': (p0[1] + p1[1]) / 2.0,
            'curvature_proxy': 0.0,
            'bbox_width': abs(dx),
            'bbox_height': abs(dy),
            'slope': 0.0,
            'nearest_point_dist': 0.0,
        }
        for ef in registry.extra_road_fields:
            vals[ef] = s['input_attrs'].get(ef, 0.0) if s.get('input_attrs') else 0.0

        feats = [float(vals.get(n, 0.0)) for n in road_names]
        edge_index.extend([[u, v], [v, u]])
        edge_attr.extend([feats, feats])
        road_records.append({'u': u, 'v': v, 'oid': s['oid'], 'geom': s['geom'], 'attrs': vals})

    return {
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'x': np.asarray(x, dtype=np.float32),
        'node_attr_maps': node_attr_maps,
        'edge_index': np.asarray(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
        'edge_attr': np.asarray(edge_attr, dtype=np.float32) if edge_attr else np.zeros((0, len(road_names)), dtype=np.float32),
        'comp': comp,
        'comp_sizes': dict(comp_sizes),
        'incident': incident,
        'road_records': road_records,
        'spatial_reference': arcpy.Describe(fc).spatialReference if arcpy else None,
        'feature_names': {'node': node_names, 'road': road_names, 'candidate': cand_names},
    }


# -------------------------------
# Candidate features / generation
# -------------------------------
def candidate_features(graph, u, v, registry: FeatureRegistry = DEFAULT_REGISTRY):
    pt_u = graph['idx_to_node'][u]
    pt_v = graph['idx_to_node'][v]
    gap = float(math.hypot(pt_u[0] - pt_v[0], pt_u[1] - pt_v[1]))
    gap_b = bearing_deg(pt_u, pt_v)
    inc_u = graph['incident'][u]
    inc_v = graph['incident'][v]
    ang_u = float(np.mean([r['bearing'] for r in inc_u])) if inc_u else 0.0
    ang_v = float(np.mean([r['bearing'] for r in inc_v])) if inc_v else 0.0
    au = angle_diff_180(ang_u, gap_b)
    av = angle_diff_180(ang_v, gap_b)
    pair_ang = angle_diff_180(ang_u, ang_v)
    deg_u = len(inc_u)
    deg_v = len(inc_v)
    len_u = [r['length'] for r in inc_u] or [0.0]
    len_v = [r['length'] for r in inc_v] or [0.0]
    comp_u = float(graph['comp_sizes'].get(graph['comp'].get(u, -1), 1))
    comp_v = float(graph['comp_sizes'].get(graph['comp'].get(v, -1), 1))
    vals = {
        'gap_len': gap,
        'gap_bearing': gap_b,
        'angle_u': ang_u,
        'angle_v': ang_v,
        'angle_to_gap_u': au,
        'angle_to_gap_v': av,
        'angle_diff': au + av,
        'comp_same': 1.0 if graph['comp'].get(u) == graph['comp'].get(v) else 0.0,
        'deg_u': float(deg_u),
        'deg_v': float(deg_v),
        'dead_u': 1.0 if deg_u <= 1 else 0.0,
        'dead_v': 1.0 if deg_v <= 1 else 0.0,
        'comp_size_u': comp_u,
        'comp_size_v': comp_v,
        'comp_size_ratio': float(min(comp_u, comp_v) / max(max(comp_u, comp_v), 1.0)),
        'degree_sum': float(deg_u + deg_v),
        'deadend_pair': float((deg_u <= 1) and (deg_v <= 1)),
        'incident_len_mean_u': float(np.mean(len_u)),
        'incident_len_mean_v': float(np.mean(len_v)),
        'incident_len_std_u': float(np.std(len_u)),
        'incident_len_std_v': float(np.std(len_v)),
        'dominant_angle_diff': pair_ang,
        'midpoint_density': 0.0,
        'detour_ratio_proxy': 0.0,
        'crossing_proxy': 0.0,
        'slope_along_gap': 0.0,
        'mask_support': 0.0,
    }
    for ef in registry.extra_candidate_fields:
        vals.setdefault(ef, 0.0)
    names = registry.all_candidate_names()
    return np.asarray([float(vals.get(n, 0.0)) for n in names], dtype=np.float32), vals


def generate_candidates(graph, registry: FeatureRegistry = DEFAULT_REGISTRY,
                        min_gap=1.0, max_gap=100.0,
                        require_dead_end=True, max_angle=60.0,
                        different_components_only=False):
    n = len(graph['idx_to_node'])
    existing = set()
    for rr in graph['road_records']:
        a, b = rr['u'], rr['v']
        existing.add((min(a, b), max(a, b)))

    cands = []
    feats = []
    metas = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in existing:
                continue
            f, m = candidate_features(graph, u, v, registry)
            gap = m['gap_len']
            if gap < min_gap or gap > max_gap:
                continue
            if require_dead_end and not (m['dead_u'] > 0.5 or m['dead_v'] > 0.5):
                continue
            if m['angle_diff'] > max_angle:
                continue
            if different_components_only and m['comp_same'] > 0.5:
                continue
            cands.append((u, v))
            feats.append(f)
            metas.append(m)

    if not cands:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, len(registry.all_candidate_names())), dtype=np.float32), []
    return np.asarray(cands, dtype=np.int64), np.asarray(feats, dtype=np.float32), metas


# -------------------------------
# Feature-class materialization
# -------------------------------
def materialize_road_features(src_fc, out_fc, registry: FeatureRegistry = DEFAULT_REGISTRY, round_digits=2, node_fc=None):
    same_path = str(src_fc).lower() == str(out_fc).lower()
    if not same_path:
        if safe_exists(out_fc):
            arcpy.management.Delete(out_fc)
        arcpy.management.CopyFeatures(src_fc, out_fc)
    graph = build_graph_from_fc(out_fc if not same_path else src_fc, registry=registry, round_digits=round_digits, node_fc=node_fc)
    out_fc = out_fc if not same_path else src_fc
    for n in graph['feature_names']['road']:
        add_field_safe(out_fc, f'RF_{n[:24]}', 'DOUBLE')
    oid_to_attrs = {rr['oid']: rr['attrs'] for rr in graph['road_records']}
    fields = ['OID@'] + [f'RF_{n[:24]}' for n in graph['feature_names']['road']]
    with arcpy.da.UpdateCursor(out_fc, fields) as cur:
        for row in cur:
            oid = row[0]
            attrs = oid_to_attrs.get(oid, {})
            for i, n in enumerate(graph['feature_names']['road'], start=1):
                row[i] = float(attrs.get(n, 0.0))
            cur.updateRow(row)
    return graph


def save_node_fc(graph, out_fc, registry: FeatureRegistry = DEFAULT_REGISTRY):
    if safe_exists(out_fc):
        arcpy.management.Delete(out_fc)
    ws, name = os.path.dirname(out_fc), os.path.basename(out_fc)
    arcpy.management.CreateFeatureclass(ws, name, 'POINT', spatial_reference=graph['spatial_reference'])
    add_field_safe(out_fc, 'NODE_ID', 'LONG')
    add_field_safe(out_fc, 'NODE_KEY', 'TEXT', 100)
    for n in graph['feature_names']['node']:
        add_field_safe(out_fc, f'NF_{n[:24]}', 'DOUBLE')
    fields = ['SHAPE@', 'NODE_ID', 'NODE_KEY'] + [f'NF_{n[:24]}' for n in graph['feature_names']['node']]
    with arcpy.da.InsertCursor(out_fc, fields) as ic:
        for i, nk in enumerate(graph['idx_to_node']):
            pt = arcpy.PointGeometry(arcpy.Point(nk[0], nk[1]), graph['spatial_reference'])
            values = [pt, i, str(nk)]
            amap = graph['node_attr_maps'][i]
            values.extend([float(amap.get(n, 0.0)) for n in graph['feature_names']['node']])
            ic.insertRow(values)


def save_candidate_fc(graph, out_fc, registry: FeatureRegistry = DEFAULT_REGISTRY,
                      min_gap=1.0, max_gap=100.0, require_dead_end=True, max_angle=60.0,
                      different_components_only=False):
    cands, feats, metas = generate_candidates(
        graph,
        registry=registry,
        min_gap=min_gap,
        max_gap=max_gap,
        require_dead_end=require_dead_end,
        max_angle=max_angle,
        different_components_only=different_components_only,
    )
    if safe_exists(out_fc):
        arcpy.management.Delete(out_fc)
    ws, name = os.path.dirname(out_fc), os.path.basename(out_fc)
    arcpy.management.CreateFeatureclass(ws, name, 'POLYLINE', spatial_reference=graph['spatial_reference'])
    add_field_safe(out_fc, 'CAND_ID', 'LONG')
    add_field_safe(out_fc, 'U_NODE', 'LONG')
    add_field_safe(out_fc, 'V_NODE', 'LONG')
    for n in registry.all_candidate_names():
        add_field_safe(out_fc, f'CF_{n[:24]}', 'DOUBLE')
    fields = ['SHAPE@', 'CAND_ID', 'U_NODE', 'V_NODE'] + [f'CF_{n[:24]}' for n in registry.all_candidate_names()]
    with arcpy.da.InsertCursor(out_fc, fields) as ic:
        for i, (u, v) in enumerate(cands.tolist() if len(cands) else []):
            p0 = graph['idx_to_node'][u]
            p1 = graph['idx_to_node'][v]
            geom = make_polyline(p0, p1, graph['spatial_reference'])
            meta = metas[i]
            row = [geom, i, int(u), int(v)] + [float(meta.get(n, 0.0)) for n in registry.all_candidate_names()]
            ic.insertRow(row)
    return cands, feats, metas


def load_prepared_candidate_fc(candidate_fc, registry: FeatureRegistry = DEFAULT_REGISTRY):
    cand_names = registry.all_candidate_names()
    field_names = ['U_NODE', 'V_NODE'] + [f'CF_{n[:24]}' for n in cand_names if field_exists(candidate_fc, f'CF_{n[:24]}')]
    pairs = []
    feats = []
    with arcpy.da.SearchCursor(candidate_fc, field_names) as cur:
        for row in cur:
            u = int(row[0])
            v = int(row[1])
            vals = list(row[2:])
            feat = []
            present = {field_names[i + 2]: vals[i] for i in range(len(vals))}
            for n in cand_names:
                feat.append(float(present.get(f'CF_{n[:24]}', 0.0) or 0.0))
            pairs.append((u, v))
            feats.append(feat)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, len(cand_names)), dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64), np.asarray(feats, dtype=np.float32)


def prepared_nodes_summary(node_fc):
    fields = [f.name for f in arcpy.ListFields(node_fc)]
    dead_field = 'NF_dead_end' if 'NF_dead_end' in fields else None
    junc_field = 'NF_junction' if 'NF_junction' in fields else None
    node_count = 0
    dead_count = 0
    junc_count = 0
    use_fields = ['OID@'] + ([dead_field] if dead_field else []) + ([junc_field] if junc_field else [])
    with arcpy.da.SearchCursor(node_fc, use_fields) as cur:
        for row in cur:
            node_count += 1
            idx = 1
            if dead_field:
                dead_count += int((row[idx] or 0) > 0.5)
                idx += 1
            if junc_field:
                junc_count += int((row[idx] or 0) > 0.5)
    return {'node_count': node_count, 'dead_end_count': dead_count, 'junction_count': junc_count}


# -------------------------------
# Training helpers
# -------------------------------
def label_candidates_from_full(train_pairs, full_graph, train_graph):
    full_edges = {(min(r['u'], r['v']), max(r['u'], r['v'])) for r in full_graph['road_records']}
    train_edges = {(min(r['u'], r['v']), max(r['u'], r['v'])) for r in train_graph['road_records']}
    y = np.asarray([
        1.0 if (min(int(u), int(v)), max(int(u), int(v))) in full_edges and (min(int(u), int(v)), max(int(u), int(v))) not in train_edges else 0.0
        for u, v in train_pairs
    ], dtype=np.float32)
    return y


def mine_hard_negatives(cand_pairs, cand_feat, cand_meta, y, neg_ratio=3, random_seed=42):
    rng = np.random.default_rng(random_seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return cand_pairs, cand_feat, y, np.arange(len(y), dtype=np.int64)

    gap_vals = np.asarray([m.get('gap_len', 0.0) for m in cand_meta], dtype=float)
    ang_vals = np.asarray([m.get('angle_diff', 999.0) for m in cand_meta], dtype=float)
    degsum_vals = np.asarray([m.get('degree_sum', 999.0) for m in cand_meta], dtype=float)
    samecomp_vals = np.asarray([m.get('comp_same', 1.0) for m in cand_meta], dtype=float)

    max_gap = np.quantile(gap_vals[pos_idx], 0.95) if len(pos_idx) > 0 else np.quantile(gap_vals, 0.75)
    hard_mask = (
        (samecomp_vals < 0.5) &
        (gap_vals <= max_gap) &
        (ang_vals <= 60.0) &
        (degsum_vals <= 4.0)
    )
    hard_neg_idx = neg_idx[hard_mask[neg_idx]]
    if len(hard_neg_idx) == 0:
        hard_neg_idx = neg_idx

    target_neg = min(len(hard_neg_idx), max(len(pos_idx), int(len(pos_idx) * neg_ratio)))
    if target_neg < len(hard_neg_idx):
        hard_neg_idx = rng.choice(hard_neg_idx, size=target_neg, replace=False)

    keep_idx = np.concatenate([pos_idx, np.asarray(hard_neg_idx, dtype=np.int64)])
    rng.shuffle(keep_idx)
    return cand_pairs[keep_idx], cand_feat[keep_idx], y[keep_idx], keep_idx


def split_train_val_indices(n, val_ratio=0.2, random_seed=42):
    rng = np.random.default_rng(random_seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0 and len(val_idx) > 0:
        train_idx = val_idx[:1]
        val_idx = val_idx[1:]
    return train_idx, val_idx


def fit_standard_scaler(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_standard_scaler(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


def precision_recall_threshold(y_true, y_prob, beta=0.5):
    thresholds = np.unique(np.asarray(y_prob, dtype=float))
    if len(thresholds) == 0:
        return 0.5, {'precision': 0.0, 'recall': 0.0, 'fbeta': 0.0}
    thresholds = np.sort(thresholds)
    best_thr = float(thresholds[-1])
    best = {'precision': 0.0, 'recall': 0.0, 'fbeta': -1.0}
    beta2 = beta * beta
    yt = np.asarray(y_true).astype(int)
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fbeta = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
        if fbeta > best['fbeta']:
            best_thr = float(thr)
            best = {'precision': float(precision), 'recall': float(recall), 'fbeta': float(fbeta)}
    return best_thr, best


# -------------------------------
# Prediction helpers
# -------------------------------
def rank_and_prune_pairs(cand_pairs, probs, cand_meta, threshold):
    rows = []
    for i, ((u, v), pr, meta) in enumerate(zip(cand_pairs, probs, cand_meta)):
        rows.append({
            'idx': i,
            'u': int(u),
            'v': int(v),
            'prob': float(pr),
            'gap_len': float(meta.get('gap_len', 0.0)),
            'angle_diff': float(meta.get('angle_diff', 999.0)),
            'comp_same': float(meta.get('comp_same', 1.0)),
            'deg_u': float(meta.get('deg_u', 99.0)),
            'deg_v': float(meta.get('deg_v', 99.0)),
        })

    rows.sort(key=lambda r: (-r['prob'], r['gap_len'], r['angle_diff']))
    used = set()
    keep_idx = []
    for r in rows:
        if r['prob'] < threshold:
            continue
        if r['comp_same'] > 0.5:
            continue
        if r['deg_u'] > 2 or r['deg_v'] > 2:
            continue
        if r['u'] in used or r['v'] in used:
            continue
        keep_idx.append(r['idx'])
        used.add(r['u'])
        used.add(r['v'])
    keep_mask = np.zeros(len(rows), dtype=np.int32)
    if keep_idx:
        keep_mask[np.asarray(keep_idx, dtype=np.int64)] = 1
    return keep_mask
