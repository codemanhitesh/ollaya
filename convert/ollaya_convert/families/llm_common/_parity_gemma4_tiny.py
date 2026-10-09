"""Validate gemma4.py Gemma4Trunk against transformers' own Gemma4UnifiedTextModel.forward on a TINY
random-init model (CPU). If the last hidden states match, the mask / position / KV-share orchestration is
correct; the 12B parity on GPU is then just confirmation."""
import os, torch

from transformers import AutoConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel
from ollaya_convert.families.llm_common.gemma4 import Gemma4Trunk

HF_TOKEN = os.environ.get("HF_TOKEN")  # only needed to read the gated base config

# real text config, then shrink the heavy dims; keep head/rope/kv-share structure
cfg = AutoConfig.from_pretrained("google/gemma-4-12b-it", token=HF_TOKEN)
tc = cfg.get_text_config()
# keep the real layer_types / head_dim / head counts / rope; shrink only the cheap dims.
tc.num_hidden_layers = 12            # includes full_attention at idx 5 and 11 (every 6th)
if hasattr(tc, "num_kv_shared_layers"):
    tc.num_kv_shared_layers = 1      # tail = [idx 11, full] -> reuses full KV stored at idx 5
tc.hidden_size = 128
tc.intermediate_size = 256
tc.vocab_size = 320
tc.sliding_window = 4
print("num_kv_shared_layers =", getattr(tc, "num_kv_shared_layers", "n/a"),
      "| layer_types[:12] =", list(tc.layer_types)[:12])

torch.manual_seed(0)
model = Gemma4UnifiedTextModel(tc).eval()
model.config._attn_implementation = "eager"

ids = torch.randint(0, tc.vocab_size, (2, 24))

with torch.no_grad():
    ref = model(input_ids=ids, use_cache=False).last_hidden_state
    trunk = Gemma4Trunk(model)
    got = trunk(ids)

diff = (ref - got).abs()
print("ref shape", tuple(ref.shape), "| got shape", tuple(got.shape))
print(f"max abs diff = {diff.max().item():.3e}   mean abs diff = {diff.mean().item():.3e}")
print("PARITY", "OK" if diff.max().item() < 1e-4 else "FAIL")
