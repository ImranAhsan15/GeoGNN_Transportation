import json
import os

import numpy as np
import torch
import torch.nn as nn

from models import CandidateScorer
from graph_utils import fit_standard_scaler, apply_standard_scaler, precision_recall_threshold, msg

EXPORT_DIR = r"C:\GIS\Project\GeoGNN_Publishable_v2\export"
EXPORT_NPZ = os.path.join(EXPORT_DIR, 'training_export.npz')
EXPORT_META = os.path.join(EXPORT_DIR, 'training_export.json')
MODEL_DIR = r"C:\GIS\Project\GeoGNN_Publishable_v2\model"
MODEL_PATH = os.path.join(MODEL_DIR, 'geognn_publishable_v2.pt')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.npz')
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 64
PATIENCE = 35
THRESHOLD_BETA = 0.5


def eval_split(model, x_t, edge_index_t, edge_attr_t, cand_pairs_t, cand_feat_t, y_np):
    model.eval()
    with torch.no_grad():
        logits = model(x_t, edge_index_t, edge_attr_t, cand_pairs_t, cand_feat_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    thr, stats = precision_recall_threshold(y_np, probs, beta=THRESHOLD_BETA)
    pred = (probs >= thr).astype(np.int32)
    acc = float((pred == y_np.astype(np.int32)).mean()) if len(y_np) else 0.0
    out = {'threshold': float(thr), 'accuracy': acc}
    out.update(stats)
    return probs, out


def main():
    arr = np.load(EXPORT_NPZ)
    with open(EXPORT_META, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    x = arr['x'].astype(np.float32)
    edge_index = arr['edge_index'].astype(np.int64)
    edge_attr = arr['edge_attr'].astype(np.float32)

    cand_pairs_train = arr['cand_pairs_train'].astype(np.int64)
    cand_feat_train = arr['cand_feat_train'].astype(np.float32)
    y_train = arr['y_train'].astype(np.float32)

    cand_pairs_val = arr['cand_pairs_val'].astype(np.int64)
    cand_feat_val = arr['cand_feat_val'].astype(np.float32)
    y_val = arr['y_val'].astype(np.float32)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError('Training requires both positive and negative training samples.')

    mu, sigma = fit_standard_scaler(cand_feat_train)
    cand_feat_train_sc = apply_standard_scaler(cand_feat_train, mu, sigma)
    cand_feat_val_sc = apply_standard_scaler(cand_feat_val, mu, sigma) if len(cand_feat_val) else cand_feat_val

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(SCALER_PATH, mean=mu, std=sigma)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    cand_pairs_train_t = torch.tensor(cand_pairs_train, dtype=torch.long, device=device)
    cand_feat_train_t = torch.tensor(cand_feat_train_sc, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    cand_pairs_val_t = torch.tensor(cand_pairs_val, dtype=torch.long, device=device)
    cand_feat_val_t = torch.tensor(cand_feat_val_sc, dtype=torch.float32, device=device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], dtype=torch.float32, device=device)

    model = CandidateScorer(x.shape[1], edge_attr.shape[1], cand_feat_train.shape[1], hidden=HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_score = -1.0
    best_val = {'threshold': 0.8, 'precision': 0.0, 'recall': 0.0, 'fbeta': 0.0, 'accuracy': 0.0}
    bad_epochs = 0

    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x_t, edge_index_t, edge_attr_t, cand_pairs_train_t, cand_feat_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        opt.step()

        if len(y_val) > 0:
            _, val_stats = eval_split(model, x_t, edge_index_t, edge_attr_t, cand_pairs_val_t, cand_feat_val_t, y_val)
            score = val_stats['fbeta']
        else:
            score = -float(loss.item())
            val_stats = {'threshold': 0.8, 'precision': 0.0, 'recall': 0.0, 'fbeta': 0.0, 'accuracy': 0.0}

        if ep % 10 == 0:
            msg(f"Epoch {ep:03d} | loss={float(loss.item()):.6f} | val_fbeta={val_stats['fbeta']:.4f} | thr={val_stats['threshold']:.4f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val = val_stats
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                msg('Early stopping.')
                break

    model.load_state_dict(best_state)
    train_probs, train_stats = eval_split(model, x_t, edge_index_t, edge_attr_t, cand_pairs_train_t, cand_feat_train_t, y_train)
    if len(y_val) > 0:
        val_probs, val_stats = eval_split(model, x_t, edge_index_t, edge_attr_t, cand_pairs_val_t, cand_feat_val_t, y_val)
    else:
        val_probs, val_stats = np.array([]), best_val

    ckpt = {
        'state_dict': model.state_dict(),
        'meta': meta,
        'dims': {
            'node': int(x.shape[1]),
            'edge': int(edge_attr.shape[1]),
            'cand': int(cand_feat_train.shape[1]),
            'hidden': HIDDEN,
        },
        'threshold': float(best_val['threshold']),
        'train_stats': train_stats,
        'val_stats': val_stats,
        'scaler_path': SCALER_PATH,
    }
    torch.save(ckpt, MODEL_PATH)

    with open(os.path.join(MODEL_DIR, 'training_report.json'), 'w', encoding='utf-8') as f:
        json.dump({'train_stats': train_stats, 'val_stats': val_stats}, f, indent=2)

    msg(f'Saved scaler: {SCALER_PATH}')
    msg(f'Saved model: {MODEL_PATH}')
    msg(f'Chosen validation threshold: {best_val["threshold"]:.6f}')
    msg(f'Train stats: {train_stats}')
    msg(f'Val stats: {val_stats}')


if __name__ == '__main__':
    main()
