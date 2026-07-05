from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: Tensor) -> Tensor:
        return _GradientReversal.apply(x, self.lambda_)


@dataclass
class ModelConfig:
    input_dim: int
    n_tissues: int
    n_disease_groups: int
    n_batches: int
    n_health_classes: int = 2
    n_sexes: int = 1
    hidden_dim: int = 256
    shared_dim: int = 32
    tissue_dim: int = 24
    disease_dim: int = 24
    nuisance_dim: int = 16
    dropout: float = 0.2
    grl_lambda: float = 1.0
    n_pathways: int = 0
    n_age_experts: int = 1
    encoder_layers: int = 3
    use_sex_condition: bool = False
    targeted_tissue_indices: tuple[int, ...] = ()
    use_joint_residual_branch: bool = False
    residual_branch_hidden_dim: int = 0
    use_separate_targeted_tissue_heads: bool = False
    use_tissue_in_age_head: bool = True
    use_disease_in_age_head: bool = False
    raw_context_dim: int = 0
    use_raw_context_in_classifiers: bool = False
    use_raw_context_in_residual: bool = False
    architecture: str = "disentangled"


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def build_encoder(
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    n_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(max(1, int(n_layers))):
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        in_dim = hidden_dim
    if layers and isinstance(layers[-1], nn.Dropout):
        layers.pop()
    return nn.Sequential(*layers)


class DisentangledAgingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_encoder(
            config.input_dim,
            config.hidden_dim,
            config.dropout,
            config.encoder_layers,
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.tissue_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.tissue_dim,
            config.dropout,
        )
        self.disease_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.disease_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        self.raw_context_dim = max(0, int(config.raw_context_dim))
        self.use_raw_context_in_classifiers = bool(
            config.use_raw_context_in_classifiers
            and self.raw_context_dim > 0
        )
        self.use_raw_context_in_residual = bool(
            config.use_raw_context_in_residual
            and self.raw_context_dim > 0
        )
        if self.raw_context_dim > 0:
            self.raw_context_proj = MLP(
                config.hidden_dim,
                max(64, config.hidden_dim // 2),
                self.raw_context_dim,
                config.dropout,
                n_layers=1,
            )
        else:
            self.raw_context_proj = None

        self.tissue_embedding = nn.Embedding(
            config.n_tissues,
            min(32, max(8, config.tissue_dim)),
        )
        self.use_sex_condition = bool(
            config.use_sex_condition and config.n_sexes > 1
        )
        if self.use_sex_condition:
            self.sex_embedding = nn.Embedding(
                config.n_sexes,
                min(8, max(3, config.shared_dim // 16)),
            )
            sex_embed_dim = self.sex_embedding.embedding_dim
        else:
            self.sex_embedding = None
            sex_embed_dim = 0
        self.n_age_experts = max(1, int(config.n_age_experts))
        self.use_joint_residual_branch = bool(
            config.use_joint_residual_branch
        )
        self.use_tissue_in_age_head = bool(
            config.use_tissue_in_age_head
        )
        self.use_disease_in_age_head = bool(
            config.use_disease_in_age_head
        )
        self.use_separate_targeted_tissue_heads = bool(
            config.use_separate_targeted_tissue_heads
        )
        self.targeted_tissue_indices = tuple(
            sorted(
                {
                    int(idx)
                    for idx in config.targeted_tissue_indices
                    if 0 <= int(idx) < config.n_tissues
                }
            )
        )
        targeted_tissue_mask = torch.zeros(
            config.n_tissues, dtype=torch.bool
        )
        if self.targeted_tissue_indices:
            targeted_tissue_mask[
                list(self.targeted_tissue_indices)
            ] = True
        self.register_buffer(
            "targeted_tissue_mask",
            targeted_tissue_mask,
            persistent=False,
        )
        age_input_dim = config.shared_dim + sex_embed_dim
        if self.use_tissue_in_age_head:
            age_input_dim += (
                config.tissue_dim
                + self.tissue_embedding.embedding_dim
            )
        if self.use_disease_in_age_head:
            age_input_dim += config.disease_dim
        residual_input_dim = age_input_dim + config.nuisance_dim
        if self.use_raw_context_in_residual:
            residual_input_dim += self.raw_context_dim
        if not self.use_tissue_in_age_head:
            residual_input_dim += (
                config.tissue_dim
                + self.tissue_embedding.embedding_dim
            )
        if not self.use_disease_in_age_head:
            residual_input_dim += config.disease_dim
        residual_hidden_dim = max(
            32,
            int(config.residual_branch_hidden_dim or 0)
            or config.hidden_dim // 3,
        )
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        if self.n_age_experts > 1:
            gate_hidden_dim = max(32, config.hidden_dim // 4)
            self.age_expert_gate = nn.Sequential(
                nn.Linear(age_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, self.n_age_experts),
            )
            self.age_expert_bias = nn.Embedding(
                config.n_tissues, self.n_age_experts
            )
            self.age_experts = nn.ModuleList(
                [
                    MLP(
                        age_input_dim,
                        max(64, config.hidden_dim // 2),
                        1,
                        config.dropout,
                        n_layers=1,
                    )
                    for _ in range(self.n_age_experts)
                ]
            )
        else:
            self.age_expert_gate = None
            self.age_expert_bias = None
            self.age_experts = None
        if self.use_joint_residual_branch:
            self.global_residual_head = nn.Sequential(
                nn.Linear(residual_input_dim, residual_hidden_dim),
                nn.GELU(),
                nn.Linear(residual_hidden_dim, 1),
            )
        else:
            self.global_residual_head = None
        if self.targeted_tissue_indices:
            targeted_input_dim = (
                residual_input_dim
                if self.use_joint_residual_branch
                else age_input_dim
            )
            if self.use_separate_targeted_tissue_heads:
                self.targeted_tissue_residual_heads = nn.ModuleDict(
                    {
                        str(idx): nn.Sequential(
                            nn.Linear(
                                targeted_input_dim,
                                residual_hidden_dim,
                            ),
                            nn.GELU(),
                            nn.Linear(residual_hidden_dim, 1),
                        )
                        for idx in self.targeted_tissue_indices
                    }
                )
                self.tissue_residual_head = None
            else:
                self.targeted_tissue_residual_heads = None
                self.tissue_residual_head = nn.Sequential(
                    nn.Linear(
                        targeted_input_dim,
                        residual_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(residual_hidden_dim, 1),
                )
        else:
            self.targeted_tissue_residual_heads = None
            self.tissue_residual_head = None
        self.shared_age_head = MLP(
            config.shared_dim, config.hidden_dim // 2, 1, config.dropout
        )
        disease_input_dim = config.disease_dim + sex_embed_dim
        if self.use_raw_context_in_classifiers:
            disease_input_dim += self.raw_context_dim
        self.health_classifier = MLP(
            disease_input_dim,
            config.hidden_dim // 2,
            config.n_health_classes,
            config.dropout,
        )
        if self.use_sex_condition:
            sex_classifier_input_dim = (
                config.shared_dim + config.tissue_dim + config.disease_dim
            )
            if self.use_raw_context_in_classifiers:
                sex_classifier_input_dim += self.raw_context_dim
            self.sex_classifier = MLP(
                sex_classifier_input_dim,
                config.hidden_dim // 2,
                config.n_sexes,
                config.dropout,
            )
        else:
            self.sex_classifier = None
        self.tissue_classifier = MLP(
            config.tissue_dim,
            config.hidden_dim // 2,
            config.n_tissues,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )

        grl_dim = (
            config.shared_dim
            + config.tissue_dim
            + config.disease_dim
        )
        self.grl = GradientReversal(config.grl_lambda)
        self.adversarial_batch_classifier = MLP(
            grl_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        self.reconstruction_head = MLP(
            config.shared_dim
            + config.tissue_dim
            + config.disease_dim
            + config.nuisance_dim,
            config.hidden_dim,
            config.input_dim,
            config.dropout,
        )
        if config.n_pathways > 0:
            self.program_head = MLP(
                config.shared_dim + config.tissue_dim,
                config.hidden_dim // 2,
                config.n_pathways,
                config.dropout,
            )
        else:
            self.program_head = None

    def forward(
        self,
        x: Tensor,
        tissue_idx: Tensor,
        sex_idx: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_tissue = self.tissue_proj(hidden)
        z_disease = self.disease_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        raw_context = (
            self.raw_context_proj(hidden)
            if self.raw_context_proj is not None
            else None
        )
        tissue_embed = self.tissue_embedding(tissue_idx)
        if self.use_sex_condition:
            if sex_idx is None:
                sex_idx = torch.zeros_like(tissue_idx)
            sex_embed = self.sex_embedding(sex_idx)
        else:
            sex_embed = None

        age_inputs = [z_shared]
        if self.use_tissue_in_age_head:
            age_inputs.extend([z_tissue, tissue_embed])
        if self.use_disease_in_age_head:
            age_inputs.append(z_disease)
        if sex_embed is not None:
            age_inputs.append(sex_embed)
        age_input = torch.cat(age_inputs, dim=1)
        disease_inputs = [z_disease]
        if sex_embed is not None:
            disease_inputs.append(sex_embed)
        if (
            self.use_raw_context_in_classifiers
            and raw_context is not None
        ):
            disease_inputs.append(raw_context)
        disease_input = torch.cat(disease_inputs, dim=1)
        total_latent = torch.cat(
            [z_shared, z_tissue, z_disease, z_nuisance], dim=1
        )
        residual_inputs = [age_input]
        if not self.use_tissue_in_age_head:
            residual_inputs.extend([z_tissue, tissue_embed])
        if not self.use_disease_in_age_head:
            residual_inputs.append(z_disease)
        residual_inputs.append(z_nuisance)
        if (
            self.use_raw_context_in_residual
            and raw_context is not None
        ):
            residual_inputs.append(raw_context)
        residual_input = torch.cat(residual_inputs, dim=1)
        grl_input = self.grl(
            torch.cat([z_shared, z_tissue, z_disease], dim=1)
        )

        base_age_pred = self.age_head(age_input).squeeze(1)
        if self.n_age_experts > 1:
            gate_logits = self.age_expert_gate(age_input)
            gate_logits = gate_logits + self.age_expert_bias(
                tissue_idx
            )
            expert_weights = torch.softmax(gate_logits, dim=1)
            expert_preds = torch.cat(
                [expert(age_input) for expert in self.age_experts],
                dim=1,
            )
            age_pred = base_age_pred + (
                expert_weights * expert_preds
            ).sum(dim=1)
        else:
            expert_weights = None
            age_pred = base_age_pred
            gate_logits = None
        core_age_pred = age_pred

        if self.global_residual_head is not None:
            joint_residual = self.global_residual_head(
                residual_input
            ).squeeze(1)
            age_pred = age_pred + joint_residual
        else:
            joint_residual = None

        if self.targeted_tissue_residual_heads is not None:
            targeted_inputs = (
                residual_input
                if self.use_joint_residual_branch
                else age_input
            )
            targeted_residual = age_pred.new_zeros(age_pred.shape)
            for idx_str, head in self.targeted_tissue_residual_heads.items():
                tissue_mask = tissue_idx.eq(int(idx_str))
                if bool(tissue_mask.any()):
                    targeted_residual[tissue_mask] = head(
                        targeted_inputs[tissue_mask]
                    ).squeeze(1)
            age_pred = age_pred + targeted_residual
        elif self.tissue_residual_head is not None:
            targeted_residual = self.tissue_residual_head(
                residual_input
                if self.use_joint_residual_branch
                else age_input
            ).squeeze(1)
            targeted_mask = self.targeted_tissue_mask[
                tissue_idx
            ].to(targeted_residual.dtype)
            age_pred = age_pred + targeted_mask * targeted_residual
        else:
            targeted_residual = None

        outputs: dict[str, Tensor] = {
            "z_shared": z_shared,
            "z_tissue": z_tissue,
            "z_disease": z_disease,
            "z_nuisance": z_nuisance,
            "age_pred": age_pred,
            "age_base_pred": base_age_pred,
            "age_core_pred": core_age_pred,
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "health_logits": self.health_classifier(disease_input),
            "tissue_logits": self.tissue_classifier(z_tissue),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
            "adversarial_batch_logits": self.adversarial_batch_classifier(
                grl_input
            ),
            "reconstruction": self.reconstruction_head(total_latent),
        }
        if expert_weights is not None:
            outputs["age_expert_weights"] = expert_weights
            outputs["age_expert_logits"] = gate_logits
        if joint_residual is not None:
            outputs["age_joint_residual"] = joint_residual
        if targeted_residual is not None:
            outputs["targeted_tissue_residual"] = targeted_residual
        if (
            joint_residual is not None
            or targeted_residual is not None
        ):
            residual_total = age_pred - core_age_pred
            outputs["age_residual_total_pred"] = residual_total
        if raw_context is not None:
            outputs["z_raw_context"] = raw_context
        if self.sex_classifier is not None:
            sex_inputs = [z_shared, z_tissue, z_disease]
            if (
                self.use_raw_context_in_classifiers
                and raw_context is not None
            ):
                sex_inputs.append(raw_context)
            outputs["sex_logits"] = self.sex_classifier(
                torch.cat(sex_inputs, dim=1)
            )
        if self.program_head is not None:
            outputs["program_scores"] = self.program_head(
                torch.cat([z_shared, z_tissue], dim=1)
            )
        return outputs


class DisentangledModelNoDisease(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.tissue_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.tissue_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        self.tissue_embedding = nn.Embedding(
            config.n_tissues,
            min(32, max(8, config.tissue_dim)),
        )
        age_input_dim = (
            config.shared_dim
            + config.tissue_dim
            + self.tissue_embedding.embedding_dim
        )
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        self.shared_age_head = MLP(
            config.shared_dim, config.hidden_dim // 2, 1, config.dropout
        )
        self.tissue_classifier = MLP(
            config.tissue_dim,
            config.hidden_dim // 2,
            config.n_tissues,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        grl_dim = config.shared_dim + config.tissue_dim
        self.grl = GradientReversal(config.grl_lambda)
        self.adversarial_batch_classifier = MLP(
            grl_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )

    def forward(
        self, x: Tensor, tissue_idx: Tensor
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_tissue = self.tissue_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        tissue_embed = self.tissue_embedding(tissue_idx)
        age_input = torch.cat(
            [z_shared, z_tissue, tissue_embed], dim=1
        )
        grl_input = self.grl(
            torch.cat([z_shared, z_tissue], dim=1)
        )
        return {
            "z_shared": z_shared,
            "z_tissue": z_tissue,
            "z_nuisance": z_nuisance,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "tissue_logits": self.tissue_classifier(z_tissue),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
            "adversarial_batch_logits": self.adversarial_batch_classifier(
                grl_input
            ),
        }


class DisentangledModelNoTissue(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.disease_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.disease_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        age_input_dim = config.shared_dim + config.disease_dim
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        self.shared_age_head = MLP(
            config.shared_dim, config.hidden_dim // 2, 1, config.dropout
        )
        self.health_classifier = MLP(
            config.disease_dim,
            config.hidden_dim // 2,
            config.n_health_classes,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        grl_dim = config.shared_dim + config.disease_dim
        self.grl = GradientReversal(config.grl_lambda)
        self.adversarial_batch_classifier = MLP(
            grl_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )

    def forward(self, x: Tensor, _tissue_idx: Tensor) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_disease = self.disease_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        age_input = torch.cat([z_shared, z_disease], dim=1)
        grl_input = self.grl(torch.cat([z_shared, z_disease], dim=1))
        return {
            "z_shared": z_shared,
            "z_disease": z_disease,
            "z_nuisance": z_nuisance,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "health_logits": self.health_classifier(z_disease),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
            "adversarial_batch_logits": self.adversarial_batch_classifier(
                grl_input
            ),
        }


class DisentangledModelNoBatch(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.tissue_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.tissue_dim,
            config.dropout,
        )
        self.disease_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.disease_dim,
            config.dropout,
        )
        self.tissue_embedding = nn.Embedding(
            config.n_tissues,
            min(32, max(8, config.tissue_dim)),
        )
        age_input_dim = (
            config.shared_dim
            + config.tissue_dim
            + self.tissue_embedding.embedding_dim
        )
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        self.shared_age_head = MLP(
            config.shared_dim, config.hidden_dim // 2, 1, config.dropout
        )
        self.health_classifier = MLP(
            config.disease_dim,
            config.hidden_dim // 2,
            config.n_health_classes,
            config.dropout,
        )
        self.tissue_classifier = MLP(
            config.tissue_dim,
            config.hidden_dim // 2,
            config.n_tissues,
            config.dropout,
        )

    def forward(
        self, x: Tensor, tissue_idx: Tensor
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_tissue = self.tissue_proj(hidden)
        z_disease = self.disease_proj(hidden)
        tissue_embed = self.tissue_embedding(tissue_idx)
        age_input = torch.cat(
            [z_shared, z_tissue, tissue_embed], dim=1
        )
        return {
            "z_shared": z_shared,
            "z_tissue": z_tissue,
            "z_disease": z_disease,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "health_logits": self.health_classifier(z_disease),
            "tissue_logits": self.tissue_classifier(z_tissue),
        }


class DisentangledModelNoOrtho(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.tissue_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.tissue_dim,
            config.dropout,
        )
        self.disease_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.disease_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        self.tissue_embedding = nn.Embedding(
            config.n_tissues,
            min(32, max(8, config.tissue_dim)),
        )
        age_input_dim = (
            config.shared_dim
            + config.tissue_dim
            + self.tissue_embedding.embedding_dim
        )
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        self.shared_age_head = MLP(
            config.shared_dim, config.hidden_dim // 2, 1, config.dropout
        )
        self.health_classifier = MLP(
            config.disease_dim,
            config.hidden_dim // 2,
            config.n_health_classes,
            config.dropout,
        )
        self.tissue_classifier = MLP(
            config.tissue_dim,
            config.hidden_dim // 2,
            config.n_tissues,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        grl_dim = (
            config.shared_dim
            + config.tissue_dim
            + config.disease_dim
        )
        self.grl = GradientReversal(config.grl_lambda)
        self.adversarial_batch_classifier = MLP(
            grl_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        self.reconstruction_head = MLP(
            config.shared_dim
            + config.tissue_dim
            + config.disease_dim
            + config.nuisance_dim,
            config.hidden_dim,
            config.input_dim,
            config.dropout,
        )
        if config.n_pathways > 0:
            self.program_head = MLP(
                config.shared_dim + config.tissue_dim,
                config.hidden_dim // 2,
                config.n_pathways,
                config.dropout,
            )
        else:
            self.program_head = None

    def forward(
        self, x: Tensor, tissue_idx: Tensor
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_tissue = self.tissue_proj(hidden)
        z_disease = self.disease_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        tissue_embed = self.tissue_embedding(tissue_idx)
        age_input = torch.cat(
            [z_shared, z_tissue, tissue_embed], dim=1
        )
        total_latent = torch.cat(
            [z_shared, z_tissue, z_disease, z_nuisance], dim=1
        )
        grl_input = self.grl(
            torch.cat([z_shared, z_tissue, z_disease], dim=1)
        )
        outputs = {
            "z_shared": z_shared,
            "z_tissue": z_tissue,
            "z_disease": z_disease,
            "z_nuisance": z_nuisance,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "health_logits": self.health_classifier(z_disease),
            "tissue_logits": self.tissue_classifier(z_tissue),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
            "adversarial_batch_logits": self.adversarial_batch_classifier(
                grl_input
            ),
            "reconstruction": self.reconstruction_head(total_latent),
        }
        if self.program_head is not None:
            outputs["program_scores"] = self.program_head(
                torch.cat([z_shared, z_tissue], dim=1)
            )
        outputs["_disable_orthogonality_loss"] = True
        return outputs


class DisentangledModelNoMonotonic(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.shared_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.shared_dim,
            config.dropout,
        )
        self.tissue_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.tissue_dim,
            config.dropout,
        )
        self.disease_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.disease_dim,
            config.dropout,
        )
        self.nuisance_proj = MLP(
            config.hidden_dim,
            config.hidden_dim,
            config.nuisance_dim,
            config.dropout,
        )
        self.tissue_embedding = nn.Embedding(
            config.n_tissues,
            min(32, max(8, config.tissue_dim)),
        )
        age_input_dim = (
            config.shared_dim
            + config.tissue_dim
            + self.tissue_embedding.embedding_dim
        )
        self.age_head = MLP(
            age_input_dim, config.hidden_dim, 1, config.dropout
        )
        self.shared_age_head = MLP(
            config.shared_dim, config.hidden_dim // 2, 1, config.dropout
        )
        self.health_classifier = MLP(
            config.disease_dim,
            config.hidden_dim // 2,
            config.n_health_classes,
            config.dropout,
        )
        self.tissue_classifier = MLP(
            config.tissue_dim,
            config.hidden_dim // 2,
            config.n_tissues,
            config.dropout,
        )
        self.nuisance_batch_classifier = MLP(
            config.nuisance_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        grl_dim = (
            config.shared_dim
            + config.tissue_dim
            + config.disease_dim
        )
        self.grl = GradientReversal(config.grl_lambda)
        self.adversarial_batch_classifier = MLP(
            grl_dim,
            config.hidden_dim // 2,
            config.n_batches,
            config.dropout,
        )
        self.reconstruction_head = MLP(
            config.shared_dim
            + config.tissue_dim
            + config.disease_dim
            + config.nuisance_dim,
            config.hidden_dim,
            config.input_dim,
            config.dropout,
        )
        if config.n_pathways > 0:
            self.program_head = MLP(
                config.shared_dim + config.tissue_dim,
                config.hidden_dim // 2,
                config.n_pathways,
                config.dropout,
            )
        else:
            self.program_head = None

    def forward(
        self, x: Tensor, tissue_idx: Tensor
    ) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        z_shared = self.shared_proj(hidden)
        z_tissue = self.tissue_proj(hidden)
        z_disease = self.disease_proj(hidden)
        z_nuisance = self.nuisance_proj(hidden)
        tissue_embed = self.tissue_embedding(tissue_idx)
        age_input = torch.cat(
            [z_shared, z_tissue, tissue_embed], dim=1
        )
        total_latent = torch.cat(
            [z_shared, z_tissue, z_disease, z_nuisance], dim=1
        )
        grl_input = self.grl(
            torch.cat([z_shared, z_tissue, z_disease], dim=1)
        )
        outputs = {
            "z_shared": z_shared,
            "z_tissue": z_tissue,
            "z_disease": z_disease,
            "z_nuisance": z_nuisance,
            "age_pred": self.age_head(age_input).squeeze(1),
            "shared_age_pred": self.shared_age_head(z_shared).squeeze(1),
            "health_logits": self.health_classifier(z_disease),
            "tissue_logits": self.tissue_classifier(z_tissue),
            "nuisance_batch_logits": self.nuisance_batch_classifier(
                z_nuisance
            ),
            "adversarial_batch_logits": self.adversarial_batch_classifier(
                grl_input
            ),
            "reconstruction": self.reconstruction_head(total_latent),
        }
        if self.program_head is not None:
            outputs["program_scores"] = self.program_head(
                torch.cat([z_shared, z_tissue], dim=1)
            )
        outputs["_disable_monotonic_loss"] = True
        return outputs
