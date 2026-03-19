import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATv2Conv
    PYG_OK = True
except Exception:
    PYG_OK = False
    GATv2Conv = None


class CandidateScorer(nn.Module):
    def __init__(self, node_in, edge_in, cand_in, hidden=64):
        super().__init__()
        self.pyg_ok = PYG_OK
        if self.pyg_ok:
            self.gat1 = GATv2Conv(node_in, hidden, heads=2, concat=True, edge_dim=edge_in)
            self.gat2 = GATv2Conv(hidden * 2, hidden, heads=1, concat=False, edge_dim=edge_in)
        else:
            self.node_proj = nn.Sequential(
                nn.Linear(node_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
        self.edge_mlp = nn.Sequential(
            nn.Linear(cand_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        if self.pyg_ok:
            h = torch.relu(self.gat1(x, edge_index, edge_attr))
            h = torch.relu(self.gat2(h, edge_index, edge_attr))
            return h
        return self.node_proj(x)

    def forward(self, x, edge_index, edge_attr, cand_pairs, cand_feat):
        h = self.encode_nodes(x, edge_index, edge_attr)
        hu = h[cand_pairs[:, 0]]
        hv = h[cand_pairs[:, 1]]
        hc = self.edge_mlp(cand_feat)
        z = torch.cat([hu, hv, torch.abs(hu - hv), hc], dim=1)
        return self.score(z).squeeze(1)
