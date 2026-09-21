from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM


EXPECTED_SEQUENCE_LENGTH = 41
EXPECTED_HANDCRAFTED_CHANNELS = 13


class DropPath(nn.Module):
    """Drop a complete residual branch independently for each sample."""

    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability must be in [0, 1)")
        self.drop_probability = float(drop_probability)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_probability == 0.0:
            return features
        keep_probability = 1.0 - self.drop_probability
        mask_shape = (features.size(0),) + (1,) * (features.ndim - 1)
        mask = torch.empty(
            mask_shape,
            dtype=features.dtype,
            device=features.device,
        ).bernoulli_(keep_probability)
        return features * mask / keep_probability


class ViewAdapter(nn.Module):
    """Project one biochemical view into the shared view-token space."""

    def __init__(self, input_dim: int, output_dim: int = 24):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected view inputs shaped [B,L,{self.input_dim}], got {tuple(inputs.shape)}"
            )
        return self.projection(inputs)


class LocalViewStemBlock(nn.Module):
    """Residual local motif extractor used independently by one view."""

    def __init__(
        self,
        channels: int = 32,
        expansion_dim: int = 96,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ):
        super().__init__()
        if channels <= 0 or expansion_dim <= 0:
            raise ValueError("channels and expansion_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)
        self.depthwise = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=self.channels,
        )
        self.expand = nn.Linear(self.channels, int(expansion_dim))
        self.contract = nn.Linear(int(expansion_dim), self.channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.size(-1) != self.channels:
            raise ValueError(
                f"Expected features shaped [B,L,{self.channels}], got {tuple(features.shape)}"
            )
        residual = features
        features = self.norm(features)
        features = self.depthwise(features.transpose(1, 2)).transpose(1, 2)
        features = self.activation(self.contract(self.expand(features)))
        return residual + self.dropout(features)


class CrossViewBiochemicalInteraction(nn.Module):
    """Attend across biochemical views independently at each sequence position."""

    def __init__(
        self,
        view_dim: int = 24,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if view_dim <= 0 or num_heads <= 0 or view_dim % num_heads != 0:
            raise ValueError("view_dim must be positive and divisible by num_heads")
        self.view_dim = int(view_dim)
        self.norm_attention = nn.LayerNorm(self.view_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.view_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(self.view_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.view_dim, self.view_dim * 2),
            nn.GELU(),
            nn.Linear(self.view_dim * 2, self.view_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.alpha_view = nn.Parameter(torch.tensor(0.1))

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 4 or views.size(2) != 4 or views.size(3) != self.view_dim:
            raise ValueError(
                "CVBI expected views shaped "
                f"[B,L,4,{self.view_dim}], got {tuple(views.shape)}"
            )
        batch_size, length, num_views, _ = views.shape
        position_views = views.reshape(batch_size * length, num_views, self.view_dim)
        normalized = self.norm_attention(position_views)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        position_views = position_views + self.alpha_view * self.attention_dropout(attended)
        position_views = position_views + self.alpha_view * self.ffn_dropout(
            self.ffn(self.norm_ffn(position_views))
        )
        return position_views.reshape(batch_size, length, num_views, self.view_dim)


class BiochemicalConditionedDynamicShortConv1D(nn.Module):
    """Per-sample, per-position, per-channel bidirectional short convolution."""

    def __init__(
        self,
        channels: int = 96,
        kernel_size: int = 5,
        dynamic_rank: int = 8,
        delta_scale: float = 0.1,
        use_global_context: bool = False,
        beta_global_init: float = 0.1,
    ):
        super().__init__()
        if channels <= 0 or dynamic_rank <= 0:
            raise ValueError("channels and dynamic_rank must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if delta_scale <= 0:
            raise ValueError("delta_scale must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dynamic_rank = int(dynamic_rank)
        self.delta_scale = float(delta_scale)
        self.use_global_context = bool(use_global_context)
        if self.use_global_context:
            self.global_projection = nn.Linear(self.channels, self.channels)
            self.beta_global = nn.Parameter(torch.tensor(float(beta_global_init)))
        else:
            self.global_projection = None
            self.register_parameter("beta_global", None)
        self.kernel_generator = nn.Sequential(
            nn.Linear(self.channels, self.dynamic_rank),
            nn.GELU(),
            nn.Linear(
                self.dynamic_rank,
                self.channels * self.kernel_size,
            ),
        )
        nn.init.zeros_(self.kernel_generator[-1].weight)
        nn.init.zeros_(self.kernel_generator[-1].bias)
        identity_kernel = torch.zeros(self.kernel_size)
        identity_kernel[self.kernel_size // 2] = 1.0
        self.register_buffer(
            "identity_kernel",
            identity_kernel.view(1, 1, 1, self.kernel_size),
            persistent=True,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        return_dynamic_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if features.ndim != 3 or features.size(-1) != self.channels:
            raise ValueError(
                f"Expected features shaped [B,L,{self.channels}], got {tuple(features.shape)}"
            )
        batch_size, length, _ = features.shape
        kernel_condition = features
        if self.use_global_context:
            global_context = self.global_projection(features.mean(dim=1)).unsqueeze(1)
            kernel_condition = features + self.beta_global * global_context
        delta_kernel = self.kernel_generator(kernel_condition).view(
            batch_size,
            length,
            self.channels,
            self.kernel_size,
        )
        kernel = self.identity_kernel.to(dtype=features.dtype) + self.delta_scale * torch.tanh(
            delta_kernel
        )
        radius = self.kernel_size // 2
        windows = F.pad(features.transpose(1, 2), (radius, radius)).unfold(
            2,
            self.kernel_size,
            1,
        )
        windows = windows.permute(0, 2, 1, 3)
        output = (windows * kernel).sum(dim=-1)
        return output, kernel if return_dynamic_weights else None


class BCDynamicShortConvBlock(nn.Module):
    """Static-dynamic hybrid local mixing followed by a SwiGLU residual FFN."""

    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 5,
        dynamic_rank: int = 16,
        dropout: float = 0.05,
        dynamic_gate_bias: float = -2.0,
        drop_path_rate: float = 0.0,
        use_global_context: bool = False,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.dynamic_norm = nn.LayerNorm(self.channels)
        self.static_conv = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=self.channels,
        )
        self.dynamic_conv = BiochemicalConditionedDynamicShortConv1D(
            channels=self.channels,
            kernel_size=kernel_size,
            dynamic_rank=dynamic_rank,
            use_global_context=use_global_context,
        )
        self.dynamic_gate = _PositionChannelGate(
            self.channels,
            initial_bias=dynamic_gate_bias,
        )
        self.dynamic_output = nn.Linear(self.channels, self.channels)
        self.dynamic_dropout = nn.Dropout(dropout)
        self.local_drop_path = DropPath(drop_path_rate)

        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn_input = nn.Linear(self.channels, self.channels * 4)
        self.ffn_output = nn.Linear(self.channels * 2, self.channels)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_drop_path = DropPath(drop_path_rate)

    def forward(
        self,
        features: torch.Tensor,
        *,
        return_dynamic_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if features.ndim != 3 or features.size(-1) != self.channels:
            raise ValueError(
                f"Expected features shaped [B,L,{self.channels}], got {tuple(features.shape)}"
            )
        normalized = self.dynamic_norm(features)
        static_delta = self.static_conv(normalized.transpose(1, 2)).transpose(1, 2)
        dynamic_delta, dynamic_weights = self.dynamic_conv(
            normalized,
            return_dynamic_weights=return_dynamic_weights,
        )
        mixed_delta = static_delta + self.dynamic_gate(normalized) * dynamic_delta
        features = features + self.local_drop_path(
            self.dynamic_dropout(self.dynamic_output(mixed_delta))
        )

        normalized = self.ffn_norm(features)
        gate, value = self.ffn_input(normalized).chunk(2, dim=-1)
        ffn_delta = self.ffn_output(F.silu(gate) * value)
        features = features + self.ffn_drop_path(self.ffn_dropout(ffn_delta))
        return features, dynamic_weights


class _PositionChannelGate(nn.Module):
    """Learn a position- and channel-specific dynamic-path contribution."""

    def __init__(self, channels: int, initial_bias: float = -2.0):
        super().__init__()
        self.channels = int(channels)
        self.projection = nn.Linear(self.channels, self.channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, float(initial_bias))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(features))


class ViewReliabilityGate(nn.Module):
    """Apply a bounded sample-adaptive residual rescaling to four views."""

    def __init__(self, view_dim: int = 32, hidden_dim: int = 32, max_delta: float = 0.2):
        super().__init__()
        if view_dim <= 0 or hidden_dim <= 0 or not 0.0 < max_delta < 1.0:
            raise ValueError("view_dim/hidden_dim must be positive and max_delta in (0, 1)")
        self.view_dim = int(view_dim)
        self.max_delta = float(max_delta)
        self.hidden = nn.Sequential(
            nn.Linear(self.view_dim * 4, int(hidden_dim)),
            nn.GELU(),
        )
        self.output = nn.Linear(int(hidden_dim), 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if views.ndim != 4 or views.size(2) != 4 or views.size(3) != self.view_dim:
            raise ValueError(
                f"Reliability gate expected [B,L,4,{self.view_dim}], got {tuple(views.shape)}"
            )
        summaries = views.mean(dim=1).flatten(start_dim=1)
        weights = 1.0 + self.max_delta * torch.tanh(
            self.output(self.hidden(summaries))
        )
        return views * weights[:, None, :, None], weights


class CVBIBCDSCHandEncoder(nn.Module):
    """Independent 13-channel handcrafted encoder with a 64-dimensional output."""

    def __init__(
        self,
        *,
        sequence_length: int = EXPECTED_SEQUENCE_LENGTH,
        view_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_view_heads: int = 4,
        dynamic_kernel_size: int = 5,
        dynamic_rank: int = 16,
        num_dynamic_blocks: int = 2,
        drop_path_rates: tuple[float, ...] | None = None,
        local_dropout: float = 0.05,
        attention_dropout: float = 0.05,
        dynamic_dropout: float = 0.05,
        output_dropout: float = 0.1,
        use_global_context: bool = False,
        use_view_reliability_gate: bool = False,
    ):
        super().__init__()
        if int(sequence_length) != EXPECTED_SEQUENCE_LENGTH:
            raise ValueError("CVBI-BCDSC currently requires sequence_length=41")
        if int(hidden_dim) != int(view_dim) * 4:
            raise ValueError("hidden_dim must equal four times view_dim")
        if num_dynamic_blocks <= 0:
            raise ValueError("num_dynamic_blocks must be positive")
        if drop_path_rates is None:
            drop_path_rates = (0.0,) * int(num_dynamic_blocks)
        if len(drop_path_rates) != int(num_dynamic_blocks):
            raise ValueError("drop_path_rates must match num_dynamic_blocks")
        self.sequence_length = int(sequence_length)
        self.view_dim = int(view_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.use_global_context = bool(use_global_context)
        self.use_view_reliability_gate = bool(use_view_reliability_gate)

        self.onehot_adapter = ViewAdapter(4, self.view_dim)
        self.chemical_adapter = ViewAdapter(4, self.view_dim)
        self.eiip_adapter = ViewAdapter(1, self.view_dim)
        self.enac_adapter = ViewAdapter(4, self.view_dim)
        self.onehot_stem = self._make_local_stem(local_dropout)
        self.chemical_stem = self._make_local_stem(local_dropout)
        self.eiip_stem = self._make_local_stem(local_dropout)
        self.enac_stem = self._make_local_stem(local_dropout)
        self.view_reliability_gate = (
            ViewReliabilityGate(self.view_dim)
            if self.use_view_reliability_gate
            else None
        )
        self.view_embedding = nn.Parameter(torch.empty(1, 1, 4, self.view_dim))
        nn.init.normal_(self.view_embedding, std=0.02)
        self.cvbi = CrossViewBiochemicalInteraction(
            view_dim=self.view_dim,
            num_heads=num_view_heads,
            dropout=attention_dropout,
        )
        self.view_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
        )
        self.dynamic_blocks = nn.ModuleList(
            BCDynamicShortConvBlock(
                channels=self.hidden_dim,
                kernel_size=dynamic_kernel_size,
                dynamic_rank=dynamic_rank,
                dropout=dynamic_dropout,
                drop_path_rate=drop_path_rates[index],
                use_global_context=self.use_global_context,
            )
            for index in range(int(num_dynamic_blocks))
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(output_dropout),
        )
        self.segment_pool = nn.AdaptiveAvgPool1d(5)
        self.pooling_projection = nn.Sequential(
            nn.Linear(self.output_dim * 6, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(output_dropout),
            nn.Linear(128, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(output_dropout),
        )

    def _make_local_stem(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            LocalViewStemBlock(
                channels=self.view_dim,
                expansion_dim=96,
                kernel_size=5,
                dropout=dropout,
            ),
            LocalViewStemBlock(
                channels=self.view_dim,
                expansion_dim=96,
                kernel_size=5,
                dropout=dropout,
            ),
        )

    def split_views(self, handcrafted_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        expected = (EXPECTED_HANDCRAFTED_CHANNELS, self.sequence_length)
        if handcrafted_features.ndim != 3 or tuple(handcrafted_features.shape[1:]) != expected:
            raise ValueError(
                "CVBI-BCDSC expected handcrafted input shaped "
                f"[B,{EXPECTED_HANDCRAFTED_CHANNELS},{self.sequence_length}], "
                f"got {tuple(handcrafted_features.shape)}"
            )
        return (
            handcrafted_features[:, 0:4, :].transpose(1, 2),
            handcrafted_features[:, 4:8, :].transpose(1, 2),
            handcrafted_features[:, 8:9, :].transpose(1, 2),
            handcrafted_features[:, 9:13, :].transpose(1, 2),
        )

    def forward(
        self,
        handcrafted_features: torch.Tensor,
        *,
        return_dynamic_weights: bool = False,
        return_view_features: bool = False,
        return_view_reliability: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        onehot, chemical, eiip, enac = self.split_views(handcrafted_features)
        views = torch.stack(
            (
                self.onehot_stem(self.onehot_adapter(onehot)),
                self.chemical_stem(self.chemical_adapter(chemical)),
                self.eiip_stem(self.eiip_adapter(eiip)),
                self.enac_stem(self.enac_adapter(enac)),
            ),
            dim=2,
        )
        stem_view_features = views
        view_reliability_weights = None
        if self.view_reliability_gate is not None:
            views, view_reliability_weights = self.view_reliability_gate(views)
        views = self.cvbi(views + self.view_embedding)
        features = self.view_fusion(
            views.reshape(views.size(0), views.size(1), self.hidden_dim)
        )
        last_dynamic_weights = None
        for index, block in enumerate(self.dynamic_blocks):
            is_last = index == len(self.dynamic_blocks) - 1
            features, dynamic_weights = block(
                features,
                return_dynamic_weights=return_dynamic_weights and is_last,
            )
            if dynamic_weights is not None:
                last_dynamic_weights = dynamic_weights
        token_features = self.output_projection(features)
        global_mean = token_features.mean(dim=1)
        segment_features = self.segment_pool(token_features.transpose(1, 2)).flatten(1)
        pooled = self.pooling_projection(
            torch.cat((global_mean, segment_features), dim=-1)
        )
        if return_dynamic_weights or return_view_features or return_view_reliability:
            details = {"features": pooled}
            if last_dynamic_weights is None:
                if return_dynamic_weights:
                    raise RuntimeError("The final dynamic block did not return diagnostic weights")
            else:
                details["dynamic_kernel_weights"] = last_dynamic_weights
            if return_view_features:
                details["view_features"] = stem_view_features
            if return_view_reliability:
                if view_reliability_weights is None:
                    raise RuntimeError("View reliability diagnostics requested without a gate")
                details["view_reliability_weights"] = view_reliability_weights
            return details
        return pooled


def _load_language_model(model_dir: Path) -> nn.Module:
    model_dir = Path(model_dir)
    required = (model_dir / "config.json", model_dir / "pytorch_model.bin")
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing language-model files in {model_dir}: {', '.join(missing)}"
        )
    config = transformers.BertConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.cls = nn.Identity()
    lora = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["Wqkv"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(model, lora)


def _mean_nucleotide_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if embeddings.ndim != 3 or embeddings.shape[:2] != attention_mask.shape:
        raise ValueError("Language-model embeddings and attention mask must align")
    active_lengths = attention_mask.long().sum(dim=1)
    content_mask = attention_mask.bool().clone()
    content_mask[:, 0] = False
    content_mask.scatter_(1, (active_lengths - 1).unsqueeze(1), False)
    content_counts = content_mask.sum(dim=1)
    if torch.any(content_counts != EXPECTED_SEQUENCE_LENGTH):
        raise ValueError(
            f"Expected {EXPECTED_SEQUENCE_LENGTH} nucleotide tokens, "
            f"got {content_counts.detach().cpu().tolist()}"
        )
    weighted = embeddings * content_mask.unsqueeze(-1)
    return weighted.sum(dim=1) / content_counts.to(embeddings.dtype).unsqueeze(1)


class HandcraftedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CVBIBCDSCHandEncoder(
            sequence_length=41,
            local_dropout=0.08,
            attention_dropout=0.05,
            dynamic_dropout=0.10,
            output_dropout=0.10,
            drop_path_rates=(0.05, 0.10),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))


class ResidualModulation(nn.Module):
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        if not 0.0 < epsilon <= 0.1:
            raise ValueError("epsilon must be in (0, 0.1]")
        self.epsilon = float(epsilon)
        self.controller = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 512),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.norm = nn.LayerNorm(256)

    def forward(
        self,
        language_features: torch.Tensor,
        handcrafted_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, beta = self.controller(handcrafted_features).chunk(2, dim=-1)
        delta = torch.tanh(gamma) * self.norm(language_features) + torch.tanh(beta)
        return language_features + self.epsilon * delta, gamma, beta


class RLMFm6APred(nn.Module):
    def __init__(self, model_dir: Path, epsilon: float = 0.05):
        super().__init__()
        self.language_model = _load_language_model(model_dir)
        hidden_size = int(getattr(self.language_model.config, "hidden_size", 768))
        self.language_projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.language_classifier = nn.Linear(256, 2)
        self.handcrafted = HandcraftedEncoder()
        self.modulation = ResidualModulation(epsilon)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def encode_language(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.language_projection(
            _mean_nucleotide_embeddings(output.logits, attention_mask)
        )

    def encode_handcrafted(self, features: torch.Tensor) -> torch.Tensor:
        return self.handcrafted.encoder(features)

    def forward_branch(self, branch: str, **inputs) -> torch.Tensor:
        if branch == "language":
            features = self.encode_language(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            return self.language_classifier(features)
        if branch == "handcrafted":
            return self.handcrafted(inputs["handcrafted_features"])
        raise ValueError(f"Unknown branch: {branch}")

    def forward(
        self,
        input_ids: torch.Tensor,
        handcrafted_features: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        language = self.encode_language(input_ids, attention_mask, token_type_ids)
        handcrafted = self.encode_handcrafted(handcrafted_features)
        fused, gamma, beta = self.modulation(language, handcrafted)
        return {
            "logits": self.classifier(fused),
            "language_logits": self.language_classifier(language),
            "handcrafted_logits": self.handcrafted.classifier(handcrafted),
            "gamma": gamma,
            "beta": beta,
        }

    def load_release_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load a compact inference-only checkpoint.

        Fixed language-model tensors are supplied by ``pretrained/`` and are
        intentionally absent from each dataset-specific fold checkpoint.
        """
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not (name.startswith("language_model.") and "lora_" not in name)
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"Checkpoint mismatch: unexpected={unexpected}, missing={missing}"
            )
