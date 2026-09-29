from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset

@dataclass
class GraphBatch:
    x: torch.Tensor
    node_type: torch.Tensor
    provenance: torch.Tensor
    missing: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    plan_tokens: torch.Tensor
    target_delta: torch.Tensor
    align_target: Optional[torch.Tensor] = None
    untreated_mask: Optional[torch.Tensor] = None
    city: str = 'unknown'
    community_id: str = 'unknown'
    intervention_family: str = 'unknown'
    metadata: Optional[Dict[str, Any]] = None

    def to(self, device):
        for f in fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v):
                setattr(self, f.name, v.to(device))
        return self

    def cpu(self):
        return self.to('cpu')

    def clone(self):
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kwargs[f.name] = v.clone() if torch.is_tensor(v) else v
        return GraphBatch(**kwargs)

class CommunityGraphDataset(Dataset):
    def __init__(self, graphs: List[GraphBatch]):
        self.graphs = graphs
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]


def save_graph(batch: GraphBatch, path: str):
    torch.save(batch, path)


def load_graph(path: str) -> GraphBatch:
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, GraphBatch): return obj
    if isinstance(obj, dict): return GraphBatch(**obj)
    raise TypeError(f'Unsupported graph object: {type(obj)}')


def fourier_position(pos: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    freq = (2.0 ** torch.arange(n_freq, device=pos.device, dtype=pos.dtype)).view(1, 1, -1)
    z = pos.unsqueeze(-1) * freq
    return torch.cat([torch.sin(z), torch.cos(z)], dim=-1).flatten(1)


def dense_adjacency(edge_index: torch.Tensor, n: int, device=None) -> torch.Tensor:
    adj = torch.zeros(n, n, dtype=torch.bool, device=device or edge_index.device)
    src, dst = edge_index
    adj[src, dst] = True
    adj[dst, src] = True
    adj.fill_diagonal_(True)
    return adj
