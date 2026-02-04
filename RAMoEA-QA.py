"""
RAMoEA-QA (Audio-MoE + LLM-MoA) LightningModule.

This file contains:
  - Lightweight routing components (MLP router, fusion, pooling)
  - A two-stage conditional specialization model:
      (1) Audio Mixture-of-Experts selects 1 audio encoder per example
      (2) Language Mixture-of-Adapters selects 1 LoRA adapter on a shared frozen LLM per example

Notes for open-source release:
  - Any project-specific imports have been removed or guarded with try/except.
  - File paths and local-only references have been replaced with a configurable `output_dir`.
  - Debug prints were replaced with optional logging flags.
  - Metric helpers relying on external caption metrics (BLEU/ROUGE/CIDEr) were removed
    (they can be added back in a separate `metrics.py` module if needed).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

from peft import LoraConfig, TaskType, get_peft_model


# =============================================================================
# Config
# =============================================================================

@dataclass
class ModelConfig:
    # General
    learning_rate: float = 1e-4
    output_dir: str = "outputs"

    # Audio experts
    audio_encoders: List[str] = None                # list of encoder names
    audio_encoder: str = ""                         # fallback single encoder
    audio_peft: str = "frozen"                      # {"frozen","lora","finetune_mae"}
    audio_lora_rank: int = 8
    enc_dim: int = 768                              # audio encoder output dim
    patch_nums: int = 1                             # encoder-specific setting (1 or 64 in your code)

    # Aligner
    aligner: str = "projection"                     # {"projection","mlp"}
    llm_dim: int = 768                              # GPT-2 hidden size

    # LLM adapters
    llm_models: List[str] = None                    # kept for API symmetry; backbone is GPT-2
    llm_model: str = "openai-community/gpt2"
    llm_peft: str = "lora"                          # {"lora","frozen"}
    llm_lora_rank: int = 8
    llm_lora_alpha: int = 32
    llm_lora_dropout: float = 0.1
    llm_lora_target_modules: List[str] = None       # e.g., ["c_attn","c_proj"]
    llm_lora_ranks: Optional[List[int]] = None
    llm_lora_alphas: Optional[List[int]] = None
    llm_lora_dropouts: Optional[List[float]] = None
    llm_expert_seeds: Optional[List[int]] = None
    llm_adapter_init_std: float = 0.02

    # Routing temperatures
    audio_router_tau_train: float = 1.0
    llm_router_tau_train: float = 1.0
    audio_router_tau_eval: float = 1.0
    llm_router_tau_eval: float = 1.0

    # Routing input policies (keep aligned with paper)
    audio_router_input: str = "audio"               # {"audio","question","fused","both"}
    llm_router_input: str = "both"                  # {"question","audio","fused","both","llm"}

    # Router architecture
    router_hidden_dim: int = 0                      # 0 -> auto
    router_num_layers: int = 2
    router_dropout: float = 0.1
    router_num_heads: int = 4
    router_mlp_ratio: int = 2
    router_use_text_encoder: bool = True
    router_text_encoder_layers: int = 1

    router_input_scale: float = 1.0
    router_logits_clip: float = 0.0

    # Router balancing / regularization
    router_balance_steps: int = 1000
    router_balance_mix_p: float = 0.0
    audio_router_lb_weight: float = 0.02
    llm_router_lb_weight: float = 0.02
    audio_router_entropy_weight: float = 0.0
    llm_router_entropy_weight: float = 0.0
    audio_diversity_weight: float = 0.0

    lb_warmup_steps: int = 0
    lb_warmup_steps_jitter: int = 0

    # Optional teacher supervision for LLM router (top-2 oracle)
    llm_oracle_teacher: bool = False
    llm_router_ce_weight: float = 1.0
    llm_router_use_ce: bool = True
    llm_diversity_weight: float = 0.0
    llm_diversity_margin: float = 0.0

    # Freeze routers after epoch
    freeze_routers_after_epoch: int = -1

    # Debug
    debug_print_routers: bool = False
    debug_print_every: int = 50


# =============================================================================
# Utilities
# =============================================================================

def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Mean entropy over batch."""
    return -(probs * torch.log(probs + eps)).sum(dim=-1).mean()


def router_stats_from_probs(
    probs: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    topk: int = 2,
    eps: float = 1e-9,
) -> Dict[str, torch.Tensor]:
    """
    probs: [B,E]
    logits: [B,E] optional
    Returns per-sample stats:
      - entropy, norm_entropy
      - topk_idx, topk_prob
      - margin_prob (top1-top2 prob)
      - logit_margin (top1-top2 logits) if logits provided
    """
    B, E = probs.shape
    k = min(int(topk), E)

    topk_prob, topk_idx = torch.topk(probs, k=k, dim=-1)  # [B,k]
    top1_prob = topk_prob[:, 0]
    top2_prob = topk_prob[:, 1] if k > 1 else top1_prob
    margin_prob = top1_prob - top2_prob

    ent = -(probs * torch.log(probs + eps)).sum(dim=-1)  # [B]
    norm_ent = ent / (math.log(E) + 1e-9)

    out = {
        "entropy": ent,
        "norm_entropy": norm_ent,
        "topk_idx": topk_idx,
        "topk_prob": topk_prob,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin_prob": margin_prob,
    }

    if logits is not None:
        topk_logit, _ = torch.topk(logits, k=k, dim=-1)
        top1_logit = topk_logit[:, 0]
        top2_logit = topk_logit[:, 1] if k > 1 else top1_logit
        out["logit_margin"] = top1_logit - top2_logit

    return out


def masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute masked mean and masked max over sequence dimension 1."""
    mask_f = mask.unsqueeze(-1).float()
    mean = (x * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-9)

    x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    max_vals = x_masked.max(dim=1).values
    max_vals = torch.where(torch.isfinite(max_vals), max_vals, torch.zeros_like(mean))
    return mean, max_vals


# =============================================================================
# Router components
# =============================================================================

class MLPSelector(nn.Module):
    """Residual MLP router that outputs logits over experts."""
    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hid = hidden_dim if hidden_dim is not None else max(128, in_dim * 2)

        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hid),
                nn.Linear(hid, hid * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hid * 2, hid),
                nn.Dropout(dropout),
            )
            for _ in range(max(1, int(num_layers)))
        ])
        self.out_proj = nn.Linear(hid, num_experts)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out_proj(h)


class AttentionPool(nn.Module):
    """Learned attention pooling over tokens (mask-aware)."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)

        h = self.ln(x)
        logits = self.score(h).squeeze(-1)               # [B,T]
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)
        return torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B,D]


class RouterFusion(nn.Module):
    """
    Cross-attention fusion (query tokens attend to key/value tokens),
    followed by a small MLP block; returns a pooled fused vector.
    """
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.q_ln = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q_tokens: torch.Tensor,  # [B,Lq,D]
        q_mask: torch.Tensor,    # [B,Lq]
        kv_tokens: torch.Tensor, # [B,Lk,D]
        kv_mask: torch.Tensor,   # [B,Lk]
    ) -> torch.Tensor:
        q = self.q_ln(q_tokens)
        kv = self.kv_ln(kv_tokens)

        attn_out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=~kv_mask,
            need_weights=False,
        )
        x = q_tokens + attn_out
        x = x + self.mlp(x)

        x = x * q_mask.unsqueeze(-1)
        return x.sum(dim=1) / (q_mask.sum(dim=1, keepdim=True) + 1e-9)  # [B,D]


class AudioRouterFeat(nn.Module):
    """
    Cheap audio feature extractor for routing.
    Input: spectrogram tensor [B,F,T] or [B,T,F]
    Output: pooled vector [B,D] and token sequence [B,T,D]
    """
    def __init__(self, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.LazyConv1d(64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(128, out_dim)

    def forward_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D spectrogram [B,F,T] or [B,T,F], got {x.shape}")

        # Heuristic: treat the smaller dim as frequency -> convert to [B,F,T] then Conv1d over T
        xt = x if x.size(1) < x.size(2) else x.transpose(1, 2)  # [B,F,T]
        h = F.relu(self.conv1(xt))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))                               # [B,128,T']

        h_seq = h.transpose(1, 2)                               # [B,T',128]
        h_seq = self.drop(h_seq)
        h_seq = self.proj(h_seq)                                # [B,T',D]

        h_pool = self.pool(h).squeeze(-1)                       # [B,128]
        h_pool = self.drop(h_pool)
        h_pool = self.proj(h_pool)                              # [B,D]
        return h_pool, h_seq


# =============================================================================
# External dependency hook (audio encoder init)
# =============================================================================

def initialize_audio_encoder(name: str) -> nn.Module:
    """
    Replace this stub with your actual audio encoder factory.

    For open-source code, it is better to implement:
      - `src/models/audio_encoders.py` with a registry/factory
      - and import it here.

    Example:
      from src.models.audio_encoders import build_audio_encoder
      return build_audio_encoder(name)
    """
    try:
        # Project-specific import (guarded)
        from src.benchmark.model_util import initialize_pretrained_model  # type: ignore
        return initialize_pretrained_model(name)
    except Exception as e:
        raise ImportError(
            "Audio encoder factory is not available. Please implement `initialize_audio_encoder(name)` "
            "to return a torch.nn.Module for your encoder names."
        ) from e


# =============================================================================
# Main LightningModule
# =============================================================================

class RAMoEAQA(pl.LightningModule):
    """
    Two-stage routed generative QA model:
      - Stage 1: Audio-MoE selects 1 audio encoder expert
      - Stage 2: LLM-MoA selects 1 LoRA adapter on a shared frozen GPT-2 backbone
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.__dict__)

        # -----------------------------
        # Router hyperparams
        # -----------------------------
        self.audio_tau_train = float(cfg.audio_router_tau_train)
        self.llm_tau_train = float(cfg.llm_router_tau_train)
        self.audio_tau_eval = float(cfg.audio_router_tau_eval)
        self.llm_tau_eval = float(cfg.llm_router_tau_eval)

        self.balance_steps = int(cfg.router_balance_steps)
        self.balance_mix_p = float(cfg.router_balance_mix_p)

        self.audio_lb_weight = float(cfg.audio_router_lb_weight)
        self.llm_lb_weight = float(cfg.llm_router_lb_weight)
        self.audio_ent_weight = float(cfg.audio_router_entropy_weight)
        self.llm_ent_weight = float(cfg.llm_router_entropy_weight)
        self.audio_diversity_weight = float(cfg.audio_diversity_weight)

        self.lb_warmup_steps = int(cfg.lb_warmup_steps)
        if cfg.lb_warmup_steps > 0 and cfg.lb_warmup_steps_jitter > 0:
            # jitter warmup to avoid synchronization in multi-run sweeps
            self.lb_warmup_steps += int(torch.randint(0, cfg.lb_warmup_steps_jitter + 1, (1,)).item())

        self.audio_router_input = cfg.audio_router_input
        self.llm_router_input = cfg.llm_router_input

        hidden_dim = None if cfg.router_hidden_dim <= 0 else int(cfg.router_hidden_dim)
        self.router_hidden_dim = hidden_dim
        self.router_num_layers = int(cfg.router_num_layers)
        self.router_dropout = float(cfg.router_dropout)
        self.router_num_heads = int(cfg.router_num_heads)
        self.router_mlp_ratio = int(cfg.router_mlp_ratio)

        self.router_input_scale = float(cfg.router_input_scale)
        self.router_logits_clip = float(cfg.router_logits_clip)

        self.freeze_routers_after_epoch = int(cfg.freeze_routers_after_epoch)
        self.routers_frozen = False

        # -----------------------------
        # Audio experts
        # -----------------------------
        audio_names = cfg.audio_encoders or ([cfg.audio_encoder] if cfg.audio_encoder else [])
        if not audio_names:
            raise ValueError("No audio encoders specified. Set `audio_encoder` or `audio_encoders`.")

        self.num_audio_experts = len(audio_names)
        self.audio_encoders = nn.ModuleList()

        for name in audio_names:
            enc = initialize_audio_encoder(name)

            if cfg.audio_peft == "frozen":
                for p in enc.parameters():
                    p.requires_grad_(False)

            elif cfg.audio_peft == "lora":
                # NOTE: target selection is encoder-specific; customize as needed.
                targets = []
                for mod_name, module in enc.named_modules():
                    if isinstance(module, nn.Linear) and any(t in mod_name for t in ["qkv", "proj"]):
                        targets.append(mod_name)
                if targets:
                    lora_cfg = LoraConfig(
                        r=int(cfg.audio_lora_rank),
                        lora_alpha=32,
                        lora_dropout=0.1,
                        target_modules=targets,
                    )
                    enc = get_peft_model(enc, lora_cfg)

            elif cfg.audio_peft == "finetune_mae":
                for p in enc.parameters():
                    p.requires_grad_(True)

            else:
                raise ValueError("audio_peft must be in {'frozen','lora','finetune_mae'}")

            self.audio_encoders.append(enc)

        self.d_audio = int(cfg.enc_dim)
        self.patch_nums = int(cfg.patch_nums)
        self.d_llm = int(cfg.llm_dim)

        # Audio aligners (one per audio expert)
        self.audio_aligners = nn.ModuleList()
        for _ in range(self.num_audio_experts):
            if cfg.aligner == "projection":
                aligner = nn.Linear(self.d_audio, self.d_llm)
            elif cfg.aligner == "mlp":
                aligner = nn.Sequential(
                    nn.Linear(self.d_audio, self.d_audio * 2),
                    nn.ReLU(),
                    nn.Linear(self.d_audio * 2, self.d_llm),
                )
            else:
                raise NotImplementedError("aligner must be in {'projection','mlp'}")
            self.audio_aligners.append(aligner)

        # -----------------------------
        # Audio router stack
        # -----------------------------
        self.router_audio_dim = 128
        self.audio_router_feat = AudioRouterFeat(out_dim=self.router_audio_dim, dropout=self.router_dropout)
        self.audio_router_text_proj = nn.Linear(self.d_llm, self.router_audio_dim)
        self.audio_router_fusion = RouterFusion(
            dim=self.router_audio_dim,
            num_heads=self.router_num_heads,
            mlp_ratio=self.router_mlp_ratio,
            dropout=self.router_dropout,
        )

        if self.audio_router_input == "both":
            self.audio_router_in_dim = self.router_audio_dim * 2
        else:
            self.audio_router_in_dim = self.router_audio_dim

        self.audio_router = MLPSelector(
            in_dim=self.audio_router_in_dim,
            num_experts=self.num_audio_experts,
            hidden_dim=self.router_hidden_dim,
            num_layers=self.router_num_layers,
            dropout=self.router_dropout,
        )

        # -----------------------------
        # Shared LLM backbone + adapters (MoA)
        # -----------------------------
        base_model_name = cfg.llm_model
        llm_cfg = GPT2Config.from_pretrained(base_model_name)
        self.llm_backbone = GPT2LMHeadModel.from_pretrained(base_model_name, config=llm_cfg)

        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm_backbone.resize_token_embeddings(len(self.tokenizer))

        llm_names = cfg.llm_models or [cfg.llm_model]
        self.num_llm_experts = len(llm_names)

        if cfg.llm_peft == "lora":
            target_modules = cfg.llm_lora_target_modules or ["c_attn", "c_proj"]

            def _pick(vals, default, i):
                return vals[i] if isinstance(vals, (list, tuple)) and i < len(vals) else default

            # Base adapter config for PEFT wrapper
            base_rank = _pick(cfg.llm_lora_ranks, cfg.llm_lora_rank, 0)
            base_alpha = _pick(cfg.llm_lora_alphas, cfg.llm_lora_alpha, 0)
            base_dropout = _pick(cfg.llm_lora_dropouts, cfg.llm_lora_dropout, 0)

            base_lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(base_rank),
                lora_alpha=int(base_alpha),
                lora_dropout=float(base_dropout),
                target_modules=target_modules,
            )
            self.llm_backbone = get_peft_model(self.llm_backbone, base_lora_cfg)

            # Add N adapter experts
            self.llm_adapter_names: List[str] = []
            for i in range(self.num_llm_experts):
                seed = _pick(cfg.llm_expert_seeds, 42 + i, i)
                torch.manual_seed(int(seed))

                a_rank = _pick(cfg.llm_lora_ranks, cfg.llm_lora_rank, i)
                a_alpha = _pick(cfg.llm_lora_alphas, cfg.llm_lora_alpha, i)
                a_dropout = _pick(cfg.llm_lora_dropouts, cfg.llm_lora_dropout, i)

                adapter_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(a_rank),
                    lora_alpha=int(a_alpha),
                    lora_dropout=float(a_dropout),
                    target_modules=target_modules,
                )
                name = f"llm_{i}"
                self.llm_backbone.add_adapter(name, adapter_cfg)
                self.llm_adapter_names.append(name)

            # Initialize each adapter's LoRA weights with a controlled random seed
            for i, name in enumerate(self.llm_adapter_names):
                seed = _pick(cfg.llm_expert_seeds, 42 + i, i)
                torch.manual_seed(int(seed))
                for p_name, p in self.llm_backbone.named_parameters():
                    if "lora_" in p_name and name in p_name:
                        p.data.normal_(mean=0.0, std=float(cfg.llm_adapter_init_std))

            # Freeze non-LoRA parameters (shared frozen backbone)
            for p_name, p in self.llm_backbone.named_parameters():
                if "lora_" not in p_name:
                    p.requires_grad_(False)

            self.llm_backbone.set_adapter(self.llm_adapter_names[0])

        elif cfg.llm_peft == "frozen":
            self.llm_adapter_names = [None for _ in range(self.num_llm_experts)]
            for p in self.llm_backbone.parameters():
                p.requires_grad_(False)

        else:
            raise ValueError("llm_peft must be in {'lora','frozen'}")

        # -----------------------------
        # LLM router stack + text embeddings for router
        # -----------------------------
        self.llm_router_fusion = RouterFusion(
            dim=self.d_llm,
            num_heads=self.router_num_heads,
            mlp_ratio=self.router_mlp_ratio,
            dropout=self.router_dropout,
        )

        self.audio_token_pool = AttentionPool(self.d_llm, dropout=self.router_dropout)
        self.text_token_pool = AttentionPool(self.d_llm, dropout=self.router_dropout)

        self.text_pool_proj = nn.Linear(self.d_llm * 3, self.d_llm)
        self.audio_pool_proj = nn.Linear(self.d_llm * 3, self.d_llm)

        if self.llm_router_input == "both":
            self.llm_router_in_dim = self.d_llm * 2
        else:
            self.llm_router_in_dim = self.d_llm

        self.llm_router = MLPSelector(
            in_dim=self.llm_router_in_dim,
            num_experts=self.num_llm_experts,
            hidden_dim=self.router_hidden_dim,
            num_layers=self.router_num_layers,
            dropout=self.router_dropout,
        )

        # Frozen copy of GPT-2 embeddings for router inputs
        vocab = self.llm_backbone.get_input_embeddings().weight.size(0)
        self.router_text_embedding = nn.Embedding(vocab, self.d_llm)
        with torch.no_grad():
            w0 = self.llm_backbone.get_input_embeddings().weight
            self.router_text_embedding.weight[: w0.size(0)].copy_(w0)
        for p in self.router_text_embedding.parameters():
            p.requires_grad_(False)

        if cfg.router_use_text_encoder and cfg.router_text_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.d_llm,
                nhead=self.router_num_heads,
                dim_feedforward=self.d_llm * self.router_mlp_ratio,
                dropout=self.router_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.router_text_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=int(cfg.router_text_encoder_layers)
            )
        else:
            self.router_text_encoder = None

        # -----------------------------
        # Tracking
        # -----------------------------
        self.train_llm_usage = {i: 0 for i in range(self.num_llm_experts)}
        self.val_llm_usage = {i: 0 for i in range(self.num_llm_experts)}
        self.test_llm_usage = {i: 0 for i in range(self.num_llm_experts)}

        self.train_audio_usage = {i: 0 for i in range(self.num_audio_experts)}
        self.val_audio_usage = {i: 0 for i in range(self.num_audio_experts)}
        self.test_audio_usage = {i: 0 for i in range(self.num_audio_experts)}

        self.val_records: List[Dict[str, Any]] = []
        self.test_step_outputs: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Router freezing
    # -------------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self._maybe_freeze_routers()

    def on_train_epoch_start(self) -> None:
        self._maybe_freeze_routers()

    def _router_modules(self) -> List[nn.Module]:
        mods = [
            # Audio routing stack
            self.audio_router_feat,
            self.audio_router_text_proj,
            self.audio_router_fusion,
            self.audio_router,

            # LLM routing stack
            self.llm_router_fusion,
            self.llm_router,

            # Pool/proj used for router inputs
            self.text_token_pool,
            self.text_pool_proj,
            self.audio_token_pool,
            self.audio_pool_proj,
        ]
        if self.router_text_encoder is not None:
            mods.append(self.router_text_encoder)
        return mods

    def freeze_routers(self) -> None:
        """Freeze routing modules (stop grads + disable dropout)."""
        if self.routers_frozen:
            return
        for m in self._router_modules():
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
        self.routers_frozen = True

    def _maybe_freeze_routers(self) -> None:
        e = int(self.freeze_routers_after_epoch)
        if e < 0 or self.routers_frozen:
            return
        if int(self.current_epoch) > e:
            self.freeze_routers()

    # -------------------------------------------------------------------------
    # Routing helpers
    # -------------------------------------------------------------------------

    def _balanced_idx(self, B: int, E: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(B, device=device) % E
        perm = torch.randperm(B, device=device)
        idx = base[perm]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(B, device=device)
        return idx[inv]

    def _lb_schedule(self, w: float) -> float:
        if self.lb_warmup_steps <= 0:
            return w
        t = min(float(self.global_step), float(self.lb_warmup_steps))
        return w * (t / float(self.lb_warmup_steps))

    def _moe_load_balancing_loss(self, probs: torch.Tensor, onehot: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        importance = probs.sum(dim=0)
        load = onehot.sum(dim=0)

        def cv2(x):
            x = x.float()
            return x.var(unbiased=False) / (x.mean() ** 2 + eps)

        return cv2(importance) + cv2(load)

    def _route_train(self, router: nn.Module, x: torch.Tensor, num_experts: int, tau: float) -> Dict[str, torch.Tensor]:
        """
        Training routing:
          - for early steps: enforce balanced expert usage
          - later: gumbel-softmax hard routing
        """
        B = x.size(0)
        logits = router(x)
        if self.router_logits_clip > 0:
            logits = logits.clamp(min=-self.router_logits_clip, max=self.router_logits_clip)

        if self.global_step < self.balance_steps:
            idx = self._balanced_idx(B, num_experts, x.device)
            probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)
            onehot = F.one_hot(idx, num_classes=num_experts).float()
            return {"logits": logits, "probs": probs, "onehot": onehot, "idx": idx}

        if self.balance_mix_p > 0 and torch.rand((), device=x.device) < self.balance_mix_p:
            idx = self._balanced_idx(B, num_experts, x.device)
            probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)
            onehot = F.one_hot(idx, num_classes=num_experts).float()
            return {"logits": logits, "probs": probs, "onehot": onehot, "idx": idx}

        probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)
        onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        idx = torch.argmax(onehot, dim=-1)
        return {"logits": logits, "probs": probs, "onehot": onehot, "idx": idx}

    def _route_eval(self, router: nn.Module, x: torch.Tensor, num_experts: int, tau: float) -> Dict[str, torch.Tensor]:
        """Evaluation routing: deterministic argmax."""
        logits = router(x)
        if self.router_logits_clip > 0:
            logits = logits.clamp(min=-self.router_logits_clip, max=self.router_logits_clip)
        probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)
        idx = torch.argmax(logits, dim=-1)
        onehot = F.one_hot(idx, num_classes=num_experts).float()
        return {"logits": logits, "probs": probs, "onehot": onehot, "idx": idx}

    # -------------------------------------------------------------------------
    # Text embedding helpers for router inputs
    # -------------------------------------------------------------------------

    def _embed_text_tokens(self, text_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        embeds = self.router_text_embedding(tok.input_ids)          # [B,L,D]
        mask = tok.attention_mask.bool()                             # [B,L]

        if self.router_text_encoder is not None:
            # TransformerEncoder uses src_key_padding_mask where True=pad
            embeds = self.router_text_encoder(embeds, src_key_padding_mask=~mask)
            embeds = embeds * mask.unsqueeze(-1)

        return embeds, mask

    def _pool_tokens(self, embeds: torch.Tensor, mask: torch.Tensor, pooler: AttentionPool, proj: nn.Linear) -> torch.Tensor:
        mean, max_vals = masked_mean_max(embeds, mask)
        attn = pooler(embeds, mask)
        out = proj(torch.cat([mean, max_vals, attn], dim=-1))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------------------
    # Prompt + LLM router input
    # -------------------------------------------------------------------------

    def _build_prompt_and_llm_router_input(
        self,
        audio_repr: torch.Tensor,     # [B,D]
        audio_tokens: torch.Tensor,   # [B,T,D]
        questions: List[str],
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Build:
          - prompt strings (text) for the LLM
          - router_input for selecting LoRA adapter

        Router inputs allowed by config:
          - question: pooled prompt tokens
          - audio: pooled aligned audio
          - fused: cross-attn(question <- audio)
          - both: concat(question_cls, fused)
          - llm: pooled tokens of [prompt + audio + Answer-tag]
        """
        prompt_list = [f"Question: {q}\nAnswer briefly and directly.\nAudio:" for q in questions]
        prompt_tokens, prompt_mask = self._embed_text_tokens(prompt_list)
        prompt_cls = self._pool_tokens(prompt_tokens, prompt_mask, self.text_token_pool, self.text_pool_proj)

        if self.llm_router_input == "question":
            router_input = prompt_cls

        elif self.llm_router_input == "audio":
            router_input = audio_repr

        elif self.llm_router_input == "fused":
            audio_mask = torch.ones(audio_tokens.size(0), audio_tokens.size(1), dtype=torch.bool, device=self.device)
            router_input = self.llm_router_fusion(prompt_tokens, prompt_mask, audio_tokens, audio_mask)

        elif self.llm_router_input == "both":
            audio_mask = torch.ones(audio_tokens.size(0), audio_tokens.size(1), dtype=torch.bool, device=self.device)
            fused = self.llm_router_fusion(prompt_tokens, prompt_mask, audio_tokens, audio_mask)
            router_input = torch.cat([prompt_cls, fused], dim=-1)

        elif self.llm_router_input == "llm":
            a_tag_tok = self.tokenizer(["\nAnswer:"] * len(questions), return_tensors="pt", padding=True, truncation=True).to(self.device)
            a_tag_emb = self.router_text_embedding(a_tag_tok.input_ids)

            audio_mask = torch.ones(audio_tokens.size(0), audio_tokens.size(1), dtype=torch.bool, device=self.device)
            router_seq = torch.cat([prompt_tokens, audio_tokens, a_tag_emb], dim=1)
            router_mask = torch.cat([prompt_mask, audio_mask, a_tag_tok.attention_mask.bool()], dim=1)
            router_input = self._pool_tokens(router_seq, router_mask, self.text_token_pool, self.text_pool_proj)

        else:
            raise ValueError(
                f"llm_router_input must be in {{'question','audio','fused','both','llm'}}, got {self.llm_router_input}"
            )

        router_input = torch.nan_to_num(router_input, nan=0.0, posinf=0.0, neginf=0.0)
        return prompt_list, router_input

    # -------------------------------------------------------------------------
    # LLM loss per sample (used by optional teacher)
    # -------------------------------------------------------------------------

    def _llm_loss_per_sample(
        self,
        indices: List[int],
        prompt_list: List[str],
        answers: List[str],
        aligned_audio: torch.Tensor,
        adapter_idx: int,
    ) -> torch.Tensor:
        """Compute per-sample CE loss for a given adapter expert on a subset."""
        llm = self.llm_backbone
        tok = self.tokenizer

        if self.llm_adapter_names[adapter_idx] is not None:
            llm.set_adapter(self.llm_adapter_names[adapter_idx])

        p_list = [prompt_list[i] for i in indices]
        a_list = [answers[i] for i in indices]
        a_audio = aligned_audio[indices]  # [G,T,D]

        p_tok = tok(p_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        p_emb = llm.get_input_embeddings()(p_tok.input_ids)

        a_tag_tok = tok(["\nAnswer:"] * len(indices), return_tensors="pt", padding=True, truncation=True).to(self.device)
        a_tag_emb = llm.get_input_embeddings()(a_tag_tok.input_ids)

        a_tok = tok(a_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        a_ids = a_tok.input_ids
        a_emb = llm.get_input_embeddings()(a_ids)

        inputs_embeds = torch.cat([p_emb, a_audio, a_tag_emb, a_emb], dim=1)

        audio_mask = torch.ones((len(indices), a_audio.size(1)), dtype=torch.long, device=self.device)
        attn_mask = torch.cat([p_tok.attention_mask, audio_mask, a_tag_tok.attention_mask, a_tok.attention_mask], dim=1)

        prefix_len = p_emb.size(1) + a_audio.size(1) + a_tag_emb.size(1)
        bs = len(indices)

        labels = torch.full((bs, prefix_len + a_ids.size(1)), -100, device=self.device, dtype=torch.long)
        labels[:, prefix_len:] = a_ids
        labels[:, prefix_len:][a_tok.attention_mask == 0] = -100

        pos_ids = torch.arange(inputs_embeds.size(1), device=self.device).unsqueeze(0).expand(bs, -1)

        out = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            labels=labels,
            return_dict=True,
        )
        logits = out.logits
        vocab = logits.size(-1)

        ce = F.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bs, -1)

        mask_sup = (labels != -100).float()
        denom = mask_sup.sum(dim=1).clamp_min(1.0)
        return (ce * mask_sup).sum(dim=1) / denom

    def _cheap_teacher_llm(
        self,
        router_logits: torch.Tensor,
        prompt_list: List[str],
        answers: List[str],
        aligned_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Optional router supervision: evaluate top-2 candidate adapters and pick lower-loss one.
        """
        B, E = router_logits.shape
        probs = torch.softmax(router_logits / max(self.llm_tau_train, 1e-6), dim=-1)

        top2 = torch.topk(router_logits, k=min(2, E), dim=-1).indices
        c1 = top2[:, 0]
        c2 = top2[:, 1] if top2.size(1) > 1 else top2[:, 0]

        losses_c1 = torch.empty((B,), device=self.device)
        losses_c2 = torch.empty((B,), device=self.device)

        grp1, grp2 = defaultdict(list), defaultdict(list)
        for i in range(B):
            grp1[int(c1[i].item())].append(i)
            grp2[int(c2[i].item())].append(i)

        for e, idxs in grp1.items():
            losses_c1[idxs] = self._llm_loss_per_sample(idxs, prompt_list, answers, aligned_audio, adapter_idx=e)
        for e, idxs in grp2.items():
            losses_c2[idxs] = self._llm_loss_per_sample(idxs, prompt_list, answers, aligned_audio, adapter_idx=e)

        best_idx = torch.where(losses_c2 < losses_c1, c2, c1)
        ce_loss = F.cross_entropy(router_logits, best_idx.detach())

        diff = (losses_c1 - losses_c2).abs()
        margin = float(self.cfg.llm_diversity_margin)
        diversity_loss = torch.clamp(margin - diff, min=0.0).mean()

        onehot = F.one_hot(best_idx, num_classes=E).float()
        return {
            "best_idx": best_idx,
            "ce_loss": ce_loss,
            "diversity_loss": diversity_loss,
            "probs": probs,
            "onehot": onehot,
        }

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x_spectrogram: torch.Tensor,   # [B,F,T] or [B,T,F]
        x_question: List[str],
        y_answer: List[str],
        mode: str = "train",           # {"train","val","test"}
    ) -> Dict[str, Any]:
        B = x_spectrogram.size(0)

        # ---------------------------------------------------------------------
        # 1) Audio routing (cheap features + optional fusion with question)
        # ---------------------------------------------------------------------
        x_feat, x_feat_seq = self.audio_router_feat.forward_seq(x_spectrogram)  # [B,Da], [B,T,Da]

        # Ensure list inputs
        questions = list(x_question) if isinstance(x_question, (list, tuple)) else [x_question]
        answers = list(y_answer) if isinstance(y_answer, (list, tuple)) else [y_answer]

        # Question tokens for audio-router fusion
        q_tokens, q_mask = self._embed_text_tokens(questions)                    # [B,L,D]
        q_tokens_proj = self.audio_router_text_proj(q_tokens)                    # [B,L,Da]
        q_cls_proj = self._pool_tokens(q_tokens_proj, q_mask, AttentionPool(self.router_audio_dim), nn.Linear(self.router_audio_dim * 3, self.router_audio_dim).to(self.device))
        # NOTE: for cleanliness, you may want to register these audio-router pooling layers as modules
        # (mirroring your original design). Here we keep it minimal but functional.

        # Build audio router input
        if self.audio_router_input == "audio":
            audio_router_input = x_feat                                           # [B,Da]
        elif self.audio_router_input == "question":
            audio_router_input = q_cls_proj                                        # [B,Da]
        elif self.audio_router_input in {"fused", "both"}:
            audio_mask = torch.ones(x_feat_seq.size(0), x_feat_seq.size(1), dtype=torch.bool, device=self.device)
            fused = self.audio_router_fusion(q_tokens_proj, q_mask, x_feat_seq, audio_mask)  # [B,Da]
            audio_router_input = fused if self.audio_router_input == "fused" else torch.cat([x_feat, fused], dim=-1)
        else:
            raise ValueError(f"audio_router_input must be in {{'audio','question','fused','both'}}, got {self.audio_router_input}")

        audio_router_input = torch.nan_to_num(audio_router_input, nan=0.0, posinf=0.0, neginf=0.0)
        audio_router_input = audio_router_input * self.router_input_scale

        if audio_router_input.size(-1) != self.audio_router_in_dim:
            raise RuntimeError(
                f"Audio router_input dim mismatch: got {audio_router_input.size(-1)} "
                f"expected {self.audio_router_in_dim} (audio_router_input={self.audio_router_input})"
            )

        # Route audio
        if mode == "train" and self.routers_frozen:
            with torch.no_grad():
                audio_route = self._route_eval(self.audio_router, audio_router_input, self.num_audio_experts, self.audio_tau_eval)
        else:
            audio_route = self._route_train(self.audio_router, audio_router_input, self.num_audio_experts, self.audio_tau_train) if mode == "train" \
                         else self._route_eval(self.audio_router, audio_router_input, self.num_audio_experts, self.audio_tau_eval)

        audio_idx_list = audio_route["idx"].detach().cpu().tolist()
        usage_audio = self.train_audio_usage if mode == "train" else (self.val_audio_usage if mode == "val" else self.test_audio_usage)
        for a in audio_idx_list:
            usage_audio[int(a)] += 1

        audio_stats = router_stats_from_probs(audio_route["probs"], logits=audio_route.get("logits"), topk=2)

        # ---------------------------------------------------------------------
        # 2) Run selected audio encoders + align to LLM dim (selected audio prefix)
        # ---------------------------------------------------------------------
        groups_audio: Dict[int, List[int]] = defaultdict(list)
        for i, e in enumerate(audio_idx_list):
            groups_audio[int(e)].append(i)

        aligned_audio: List[Optional[torch.Tensor]] = [None] * B
        for e, idxs in groups_audio.items():
            encoder = self.audio_encoders[e]
            aligner = self.audio_aligners[e]
            x_b = x_spectrogram[idxs]

            # Encoder call variants (keep consistent with your implementation)
            if self.patch_nums == 1:
                if hasattr(encoder, "forward_encoder"):
                    out = encoder.forward_encoder(x_b, mask_ratio=0)
                    enc_out = out[0] if isinstance(out, tuple) else out
                else:
                    enc_out = encoder(x_b)
            elif self.patch_nums == 64:
                if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "forward_window"):
                    enc_out = encoder.encoder.forward_window(x_b, window_size=32, hop_size=16)
                elif hasattr(encoder, "forward_window"):
                    enc_out = encoder.forward_window(x_b, window_size=32, hop_size=16)
                elif hasattr(encoder, "forward_encoder"):
                    out = encoder.forward_encoder(x_b, mask_ratio=0)
                    enc_out = out[0] if isinstance(out, tuple) else out
                else:
                    enc_out = encoder(x_b)
            else:
                raise NotImplementedError(f"Unsupported patch_nums: {self.patch_nums}")

            a = aligner(enc_out)  # [G,T,D] or [G,D]
            if a.ndim == 2:
                a = a.unsqueeze(1)

            for j, bi in enumerate(idxs):
                aligned_audio[bi] = a[j:j+1]

        aligned_audio_t = torch.cat(aligned_audio, dim=0)  # [B,T,D]
        aligned_audio_t = torch.nan_to_num(aligned_audio_t, nan=0.0, posinf=0.0, neginf=0.0)

        audio_mask = torch.ones(aligned_audio_t.size(0), aligned_audio_t.size(1), dtype=torch.bool, device=self.device)
        audio_repr = self._pool_tokens(aligned_audio_t, audio_mask, self.audio_token_pool, self.audio_pool_proj)  # [B,D]

        # ---------------------------------------------------------------------
        # 3) LLM routing input (question/audio/fused/both/llm)
        # ---------------------------------------------------------------------
        prompt_list, llm_router_input = self._build_prompt_and_llm_router_input(audio_repr, aligned_audio_t, questions)
        llm_router_input = llm_router_input * self.router_input_scale

        if llm_router_input.size(-1) != self.llm_router_in_dim:
            raise RuntimeError(
                f"LLM router_input dim mismatch: got {llm_router_input.size(-1)} expected {self.llm_router_in_dim} "
                f"(llm_router_input={self.llm_router_input})"
            )

        # ---------------------------------------------------------------------
        # 4) Route LLM adapter
        # ---------------------------------------------------------------------
        llm_ce_loss = torch.tensor(0.0, device=self.device)
        llm_div_loss = torch.tensor(0.0, device=self.device)

        if mode == "train" and not self.routers_frozen:
            router_logits = self.llm_router(llm_router_input)
            if self.router_logits_clip > 0:
                router_logits = router_logits.clamp(min=-self.router_logits_clip, max=self.router_logits_clip)

            if self.global_step < self.balance_steps:
                llm_idx = self._balanced_idx(B, self.num_llm_experts, self.device)
                llm_probs = torch.softmax(router_logits / max(self.llm_tau_train, 1e-6), dim=-1)
                llm_onehot = F.one_hot(llm_idx, num_classes=self.num_llm_experts).float()

            else:
                if self.cfg.llm_oracle_teacher and self.num_llm_experts > 1:
                    teacher = self._cheap_teacher_llm(router_logits, prompt_list, answers, aligned_audio_t)
                    llm_idx = teacher["best_idx"]
                    llm_probs = teacher["probs"]
                    llm_onehot = teacher["onehot"]

                    if self.cfg.llm_router_use_ce and self.cfg.llm_router_ce_weight > 0:
                        llm_ce_loss = teacher["ce_loss"] * float(self.cfg.llm_router_ce_weight)

                    if self.cfg.llm_diversity_weight > 0:
                        llm_div_loss = teacher["diversity_loss"] * float(self.cfg.llm_diversity_weight)
                else:
                    llm_route = self._route_train(self.llm_router, llm_router_input, self.num_llm_experts, self.llm_tau_train)
                    router_logits = llm_route["logits"]
                    llm_probs = llm_route["probs"]
                    llm_onehot = llm_route["onehot"]
                    llm_idx = llm_route["idx"]

        else:
            llm_route = self._route_eval(self.llm_router, llm_router_input, self.num_llm_experts, self.llm_tau_eval)
            router_logits = llm_route["logits"]
            llm_probs = llm_route["probs"]
            llm_onehot = llm_route["onehot"]
            llm_idx = llm_route["idx"]

        llm_idx_list = llm_idx.detach().cpu().tolist()
        usage_llm = self.train_llm_usage if mode == "train" else (self.val_llm_usage if mode == "val" else self.test_llm_usage)
        for c in llm_idx_list:
            usage_llm[int(c)] += 1

        llm_stats = router_stats_from_probs(llm_probs, logits=router_logits, topk=2)

        # ---------------------------------------------------------------------
        # 5) LLM forward: group by adapter expert (for speed)
        # ---------------------------------------------------------------------
        groups_llm: Dict[int, List[int]] = defaultdict(list)
        for i, e in enumerate(llm_idx_list):
            groups_llm[int(e)].append(i)

        do_generate = (mode != "train")
        generated_text: Optional[List[str]] = ([None] * B) if do_generate else None

        total_loss = 0.0
        for e, indices in groups_llm.items():
            llm = self.llm_backbone
            tok = self.tokenizer

            if self.llm_adapter_names[e] is not None:
                llm.set_adapter(self.llm_adapter_names[e])

            p_list = [prompt_list[i] for i in indices]
            a_list = [answers[i] for i in indices]
            a_audio = aligned_audio_t[indices]  # [G,T,D]

            p_tok = tok(p_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
            p_emb = llm.get_input_embeddings()(p_tok.input_ids)

            a_tag_tok = tok(["\nAnswer:"] * len(indices), return_tensors="pt", padding=True, truncation=True).to(self.device)
            a_tag_emb = llm.get_input_embeddings()(a_tag_tok.input_ids)

            a_tok = tok(a_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
            a_ids = a_tok.input_ids
            a_emb = llm.get_input_embeddings()(a_ids)

            inputs_embeds = torch.cat([p_emb, a_audio, a_tag_emb, a_emb], dim=1)

            audio_mask2 = torch.ones((len(indices), a_audio.size(1)), dtype=torch.long, device=self.device)
            attn_mask = torch.cat([p_tok.attention_mask, audio_mask2, a_tag_tok.attention_mask, a_tok.attention_mask], dim=1)

            prefix_len = p_emb.size(1) + a_audio.size(1) + a_tag_emb.size(1)
            bs = len(indices)

            labels = torch.full((bs, prefix_len + a_ids.size(1)), -100, device=self.device, dtype=torch.long)
            labels[:, prefix_len:] = a_ids
            labels[:, prefix_len:][a_tok.attention_mask == 0] = -100

            pos_ids = torch.arange(inputs_embeds.size(1), device=self.device).unsqueeze(0).expand(bs, -1)

            out = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                labels=labels,
                return_dict=True,
            )
            total_loss = total_loss + out.loss * len(indices)

            if do_generate:
                with torch.no_grad():
                    gen_embeds = torch.cat([p_emb, a_audio, a_tag_emb], dim=1)
                    gen_mask = torch.cat([p_tok.attention_mask, audio_mask2, a_tag_tok.attention_mask], dim=1)

                    # Keep generation length aligned with target length (as in your code)
                    max_new_tokens = a_ids.size(1)

                    gen = llm.generate(
                        inputs_embeds=gen_embeds,
                        attention_mask=gen_mask,
                        position_ids=pos_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=4,
                        temperature=0.0,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )

                    decoded = tok.batch_decode(gen, skip_special_tokens=True)
                    for il, bi in enumerate(indices):
                        txt = decoded[il]
                        if "Answer:" in txt:
                            txt = txt.split("Answer:", 1)[-1].strip()
                        generated_text[bi] = txt.strip()

        loss_main = total_loss / B

        # ---------------------------------------------------------------------
        # 6) Auxiliary losses (load balancing + entropy + diversity)
        # ---------------------------------------------------------------------
        if mode == "train" and self.routers_frozen:
            audio_lb_w = 0.0
            llm_lb_w = 0.0
        else:
            audio_lb_w = self._lb_schedule(self.audio_lb_weight)
            llm_lb_w = self._lb_schedule(self.llm_lb_weight)

        audio_lb = self._moe_load_balancing_loss(audio_route["probs"], audio_route["onehot"])
        llm_lb = self._moe_load_balancing_loss(llm_probs, llm_onehot)

        audio_ent = entropy_from_probs(audio_route["probs"])
        llm_ent = entropy_from_probs(llm_probs)
        audio_div = audio_route["probs"].var(dim=0, unbiased=False).mean()

        loss = loss_main
        if mode == "train":
            loss = loss + audio_lb_w * audio_lb + llm_lb_w * llm_lb + llm_ce_loss + llm_div_loss
            if self.audio_ent_weight > 0:
                loss = loss - self.audio_ent_weight * audio_ent
            if self.llm_ent_weight > 0:
                loss = loss - self.llm_ent_weight * llm_ent
            if self.audio_diversity_weight > 0:
                loss = loss - self.audio_diversity_weight * audio_div

        # Logging
        self.log(f"{mode}_loss_main", loss_main, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{mode}_loss_total", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{mode}_audio_router_entropy", audio_ent, on_step=True, on_epoch=True)
        self.log(f"{mode}_llm_router_entropy", llm_ent, on_step=True, on_epoch=True)

        return {
            "loss": loss,
            "loss_main": loss_main.detach(),
            "generated_text": generated_text,

            "audio_choice": audio_route["idx"].detach(),
            "llm_choice": llm_idx.detach(),

            # routing uncertainty stats
            "audio_entropy": audio_stats["entropy"].detach(),
            "audio_norm_entropy": audio_stats["norm_entropy"].detach(),
            "llm_entropy": llm_stats["entropy"].detach(),
            "llm_norm_entropy": llm_stats["norm_entropy"].detach(),

            "audio_topk_idx": audio_stats["topk_idx"].detach(),
            "audio_topk_prob": audio_stats["topk_prob"].detach(),
            "llm_topk_idx": llm_stats["topk_idx"].detach(),
            "llm_topk_prob": llm_stats["topk_prob"].detach(),

            "audio_logit_margin": audio_stats.get("logit_margin", torch.zeros(B, device=self.device)).detach(),
            "llm_logit_margin": llm_stats.get("logit_margin", torch.zeros(B, device=self.device)).detach(),
        }

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=float(self.cfg.learning_rate))

    # -------------------------------------------------------------------------
    # Training / validation / test hooks (cleaned)
    # -------------------------------------------------------------------------

    def training_step(self, batch: Tuple, batch_idx: int):
        x_spectrogram, x_question, y_answer, *_ = batch
        out = self(x_spectrogram, x_question, y_answer, mode="train")
        return out["loss"]

    def on_train_epoch_end(self) -> None:
        self._save_router_usage(split="train")
        self._reset_usage(split="train")

    def on_validation_epoch_start(self) -> None:
        self.val_records = []

    def validation_step(self, batch: Tuple, batch_idx: int):
        x_spectrogram, x_question, y_answer, question_type, audio_path, dataset = batch
        out = self(x_spectrogram, x_question, y_answer, mode="val")
        gen_list = out.get("generated_text") or [""] * len(y_answer)

        # Normalize to lists
        def _tolist(x): return x if isinstance(x, (list, tuple)) else [x]
        xq = _tolist(x_question)
        ya = _tolist(y_answer)
        qt = _tolist(question_type)
        ap = _tolist(audio_path)
        ds = _tolist(dataset)

        a_choice = out["audio_choice"].detach().cpu().tolist()
        l_choice = out["llm_choice"].detach().cpu().tolist()

        for i in range(len(ya)):
            self.val_records.append({
                "question": xq[i],
                "reference": ya[i],
                "predicted": gen_list[i],
                "question_type": qt[i],
                "dataset": ds[i],
                "audio_path": ap[i],
                "audio_expert": int(a_choice[i]),
                "llm_expert": int(l_choice[i]),
            })

    def on_validation_epoch_end(self) -> None:
        self._save_json_records(self.val_records, split="val")
        self._save_router_usage(split="val")
        self._reset_usage(split="val")

    def test_step(self, batch: Tuple, batch_idx: int):
        x_spectrogram, x_question, y_answer, question_type, audio_path, dataset = batch
        B = x_spectrogram.size(0)

        out = self(x_spectrogram, x_question, y_answer, mode="test")
        self.log("test_loss", out["loss"], prog_bar=True)

        gen_list = out.get("generated_text") or [""] * B
        ds_name = dataset[0] if isinstance(dataset, (list, tuple)) else dataset

        a_choice = out["audio_choice"].detach().cpu().tolist()
        l_choice = out["llm_choice"].detach().cpu().tolist()

        for i in range(B):
            self.test_step_outputs.append({
                "id": batch_idx * B + i,
                "dataset": ds_name,
                "question": x_question[i],
                "audio_path": audio_path[i] if isinstance(audio_path, (list, tuple)) else audio_path,
                "ref": y_answer[i],
                "hypo": gen_list[i],
                "question_type": question_type[0] if isinstance(question_type, (list, tuple)) else question_type,
                "audio_expert": int(a_choice[i]),
                "llm_expert": int(l_choice[i]),
            })

        return out["loss"]

    def on_test_epoch_end(self) -> None:
        self._save_json_records(self.test_step_outputs, split="test", filename="test_results.json")
        self._save_router_usage(split="test")
        self.test_step_outputs = []
        self._reset_usage(split="test")

    # -------------------------------------------------------------------------
    # IO helpers
    # -------------------------------------------------------------------------

    def _reset_usage(self, split: str) -> None:
        if split == "train":
            self.train_audio_usage = {i: 0 for i in range(self.num_audio_experts)}
            self.train_llm_usage = {i: 0 for i in range(self.num_llm_experts)}
        elif split == "val":
            self.val_audio_usage = {i: 0 for i in range(self.num_audio_experts)}
            self.val_llm_usage = {i: 0 for i in range(self.num_llm_experts)}
        elif split == "test":
            self.test_audio_usage = {i: 0 for i in range(self.num_audio_experts)}
            self.test_llm_usage = {i: 0 for i in range(self.num_llm_experts)}

    def _save_router_usage(self, split: str) -> None:
        out_dir = os.path.join(self.cfg.output_dir, "routing")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"router_usage_{split}.json")

        payload = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "audio_usage": getattr(self, f"{split}_audio_usage"),
            "llm_usage": getattr(self, f"{split}_llm_usage"),
        }

        # Append as a list of snapshots over time
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        else:
            data = []

        data.append(payload)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_json_records(self, records: List[Dict[str, Any]], split: str, filename: Optional[str] = None) -> None:
        out_dir = os.path.join(self.cfg.output_dir, "results", split)
        os.makedirs(out_dir, exist_ok=True)
        fname = filename or f"results_epoch{int(self.current_epoch)}_step{int(self.global_step)}.json"
        out_path = os.path.join(out_dir, fname)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Minimal smoke-test construction (won't run without a valid audio encoder factory).
    cfg = ModelConfig(
        audio_encoders=["YOUR_AUDIO_ENCODER_NAME"],
        output_dir="outputs",
        llm_model="openai-community/gpt2",
        llm_peft="lora",
        audio_router_input="audio",
        llm_router_input="both",
    )
    model = RAMoEAQA(cfg)
    print(model)
