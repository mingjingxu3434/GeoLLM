import torch
import torch.nn as nn

class PlanningLanguageAdapter(nn.Module):
    """Compact 6-layer Transformer adapter producing <=16 structured intervention tokens.

    The paper specifies field semantics but not a released tokenizer/parser. This reproduction
    uses a trainable token embedding + Transformer encoder + learned query pooling. Field heads
    make the representation explicit and can be supervised when annotations are available.
    """
    def __init__(self, vocab_size, d_model=768, nhead=12, layers=6, ffn=3072,
                 max_tokens=16, out_dim=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.queries = nn.Parameter(torch.randn(max_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.to_intervention = nn.Linear(d_model, out_dim)
        self.action_head = nn.Linear(d_model, 8)
        self.target_type_head = nn.Linear(d_model, 4)
        self.magnitude_head = nn.Linear(d_model, 2)
        self.constraint_head = nn.Linear(d_model, 3)

    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        z = self.encoder(self.embed(token_ids))
        q = self.queries.unsqueeze(0).expand(z.size(0), -1, -1)
        pooled, _ = self.cross_attn(q, z, z, need_weights=False)
        intervention = self.to_intervention(pooled)
        fields = {
            'action_logits': self.action_head(pooled),
            'target_type_logits': self.target_type_head(pooled),
            'magnitude': torch.sigmoid(self.magnitude_head(pooled)[..., :1]),
            'horizon': torch.sigmoid(self.magnitude_head(pooled)[..., 1:]),
            'constraint_logits': self.constraint_head(pooled),
        }
        return intervention, fields
