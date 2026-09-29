from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    node_feat_dim: int = 24
    edge_feat_dim: int = 8
    hidden_dim: int = 256
    edge_hidden_dim: int = 64
    node_type_count: int = 4
    provenance_count: int = 4
    language_vocab_size: int = 4096
    language_dim: int = 768
    language_layers: int = 6
    language_heads: int = 12
    language_ffn_dim: int = 3072
    max_plan_len: int = 384
    max_intervention_tokens: int = 16
    intervention_dim: int = 256
    operator_layers: int = 8
    operator_heads: int = 8
    operator_ffn_dim: int = 1024
    hyper_rank: int = 8
    lora_rank: int = 16
    dropout: float = 0.10
    num_targets: int = 4
    use_language_operator: bool = True
    use_direct_update: bool = True
    use_uncertainty_head: bool = True
    static_plan_conditioning: bool = False

    def to_dict(self):
        return asdict(self)

@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 1
    spatial_lr: float = 2e-4
    language_lr: float = 5e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    patience: int = 15
    lambda_align: float = 0.5
    lambda_cons: float = 0.25
    lambda_para: float = 0.10
    lambda_dir: float = 0.05
    seed: int = 7
    mixed_precision: bool = True
    gradient_accumulation: int = 4
    coverage: float = 0.90

    def to_dict(self):
        return asdict(self)
