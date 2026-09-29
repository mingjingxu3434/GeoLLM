import torch

def spatial_block_corrupt(batch, fraction: float, seed: int = 0):
    """Mask a spatially contiguous block of noncritical attributes.

    Geometry, node identity, and intervention text are preserved. Corruption selects nodes
    closest to a random anchor and masks a fraction of feature entries on those nodes.
    """
    if fraction <= 0: return batch.clone()
    g = batch.clone(); gen = torch.Generator().manual_seed(seed)
    n = g.x.size(0)
    anchor = int(torch.randint(0, n, (1,), generator=gen))
    dist = ((g.pos - g.pos[anchor]) ** 2).sum(-1)
    k = max(1, int(round(n * min(1.0, fraction))))
    nodes = torch.topk(dist, k, largest=False).indices
    # mask 50% of feature columns on selected nodes; preserves geometric position itself
    f = g.x.size(1); cols = torch.randperm(f, generator=gen)[:max(1, f//2)]
    g.x[nodes[:, None], cols[None, :]] = 0.0
    g.missing[nodes[:, None], cols[None, :]] = 1.0
    meta = dict(g.metadata or {}); meta['corruption_fraction'] = fraction; g.metadata = meta
    return g
