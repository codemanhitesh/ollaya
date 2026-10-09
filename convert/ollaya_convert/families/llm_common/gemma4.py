"""Gemma 4 (`gemma4_unified` text model) recompute for the ONNX head path, the sibling of `gemma3.py`.

Same idea: re-express the HF forward as a single cache-free prefill that ONNX can run, reusing the HF
submodules (scaled embedding, the decoder layers with their norms / `v_norm` / KV sharing / `layer_scalar` /
MLP, the rotary module, the final norm) so the math is transformers' own. Only the two attention masks are
built here, as additive float masks, instead of transformers' `create_*_mask` helpers.

Gemma 4 vs Gemma 3 (all handled by reusing the real submodules, not reimplemented here):
  * split heads by layer type (sliding `head_dim`, 8 KV heads; full `global_head_dim`, 1 KV head, K=V shared);
  * dual RoPE: a table per layer type from one `rotary_emb` module, called `rotary_emb(x, position_ids, lt)`;
  * `q_norm` / `k_norm` / `v_norm` per head, `scaling = 1.0`, no attention soft-capping;
  * per-layer `layer_scalar`, and a double-wide MLP on the KV-shared tail;
  * KV sharing: the tail layers reuse the K/V of the last storing layer via `shared_kv_states` (a dict the
    decoder layers fill and read themselves — we just thread one empty dict through, as the HF forward does).

This mirrors `Gemma4UnifiedTextModel.forward` (cache-free, `use_cache=False`) and returns the last hidden
state after the final norm. The attention implementation is forced to eager so the additive masks apply.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from collections import UserDict
except Exception:  # pragma: no cover
    UserDict = dict


class Gemma4Trunk(nn.Module):
    """input_ids [B, T] -> last hidden state [B, T, hidden] (after the final norm)."""

    def __init__(self, text_model):
        super().__init__()
        self.m = text_model
        cfg = text_model.config
        if getattr(cfg, "attn_logit_softcapping", None):
            raise ValueError("attention soft-capping is not supported by this recompute")
        # eager attention so our additive float masks are the ones applied
        cfg._attn_implementation = "eager"
        self.layer_types = list(cfg.layer_types)[: cfg.num_hidden_layers]
        self.unique_layer_types = sorted(set(self.layer_types))
        self.window = int(cfg.sliding_window)

    def _masks(self, T, dtype, device):
        """Additive [1, 1, T, T] masks: global = causal, sliding = causal within the window."""
        neg = torch.finfo(dtype).min
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        causal_ok = j <= i                                  # key j visible from query i
        sliding_ok = causal_ok & (i - j < self.window)      # ... and within the sliding window
        masks = {}
        for lt in self.unique_layer_types:
            ok = sliding_ok if lt == "sliding_attention" else causal_ok
            m = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(~ok, neg)
            masks[lt] = m[None, None, :, :]
        return masks

    def forward(self, input_ids):
        m = self.m
        x = m.embed_tokens(input_ids)                       # scaled word embedding
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device)[None, :].expand(B, -1)

        pe = {lt: m.rotary_emb(x, position_ids, lt) for lt in self.unique_layer_types}
        masks = self._masks(T, x.dtype, x.device)

        shared_kv_states = UserDict()
        for i, layer in enumerate(m.layers[: len(self.layer_types)]):
            lt = self.layer_types[i]
            x = layer(
                x,
                shared_kv_states=shared_kv_states,
                position_embeddings=pe[lt],
                attention_mask=masks[lt],
                position_ids=position_ids,
            )
        return m.norm(x)
