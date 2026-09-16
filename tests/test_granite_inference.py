"""Debug trace of Granite 4.0-H-350M inference.

Registers forward hooks on every nn.Module to capture the full call graph
during a single forward pass. Produces a structured trace file mapping each
Python function call to its corresponding hardware unit in src/.
"""

import time
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "granite-4.0-h-350m"
TRACE_OUT = Path(__file__).resolve().parent.parent / "inference_trace.json"

# Map PyTorch module class names to hardware unit directories
UNIT_MAP = {
    "Embedding": "embedding_lookup_unit",
    "GraniteMoeHybridRMSNorm": "RMSNorm_unit",
    "GraniteMoeHybridRMSNormGated": "RMSNorm_unit",
    "GraniteMoeHybridMambaLayer": "mamba2_unit",
    "GraniteMoeHybridAttention": "attention_unit",
    "GraniteMoeHybridMLP": "mlp_unit",
    "Linear": "matrix_unit",
    "Conv1d": "conv1d_unit",
    "SiLUActivation": "SiLU_unit",
    "GraniteMoeHybridForCausalLM": None,
    "GraniteMoeHybridModel": None,
    "GraniteMoeHybridDecoderLayer": None,
}


class ModuleTracer:
    """Collects forward-pass traces via PyTorch module hooks."""

    def __init__(self):
        self.entries: list[dict] = []
        self._hooks = []
        self._pending: dict[int, dict] = {}

    def _pre_hook(self, module, inputs):
        module_id = id(module)
        name = self._get_name(module)
        cls_name = module.__class__.__name__

        entry = {
            "name": name,
            "class": cls_name,
            "hw_unit": UNIT_MAP.get(cls_name, "unknown"),
            "start": time.perf_counter(),
            "input_shapes": self._shapes(inputs),
            "output_shapes": [],
        }
        self.entries.append(entry)
        self._pending[module_id] = entry

    def _post_hook(self, module, inputs, outputs):
        entry = self._pending.pop(id(module), None)
        if entry is None:
            return
        entry["elapsed_ms"] = round((time.perf_counter() - entry["start"]) * 1000, 3)
        entry["output_shapes"] = self._shapes(outputs)
        del entry["start"]

    def _get_name(self, module) -> str:
        for name, mod in module.named_modules():
            if id(mod) == id(module) and name:
                return name
        return module.__class__.__name__

    def _shapes(self, tensors) -> list[str]:
        if isinstance(tensors, torch.Tensor):
            return [self._fmt_tensor(tensors)]
        if isinstance(tensors, (tuple, list)):
            return [self._fmt_tensor(t) if isinstance(t, torch.Tensor)
                    else ("None" if t is None else type(t).__name__)
                    for t in tensors]
        return [type(tensors).__name__]

    def _fmt_tensor(self, t: torch.Tensor) -> str:
        shape = "x".join(str(d) for d in t.shape)
        return f"{shape} {t.dtype}"

    def register(self, model: torch.nn.Module):
        for module in model.modules():
            pre = module.register_forward_pre_hook(self._pre_hook)
            post = module.register_forward_hook(self._post_hook)
            self._hooks.extend([pre, post])

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def to_json(self) -> list[dict]:
        """Return trace entries as a clean list for JSON export."""
        return [
            {
                "step": i,
                "module": e["name"],
                "class": e["class"],
                "hw_unit": e["hw_unit"],
                "input": e["input_shapes"],
                "output": e["output_shapes"],
                "ms": e["elapsed_ms"],
            }
            for i, e in enumerate(self.entries)
        ]


def main():
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
    )
    model.eval()

    # Set up tracing
    tracer = ModuleTracer()
    tracer.register(model)

    # Run inference
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input: {prompt}")
    print(f"Tokens: {input_ids.tolist()}")

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(input_ids)
        total_ms = (time.perf_counter() - t0) * 1000

    logits = outputs.logits
    predicted_id = logits[0, -1].argmax().item()
    predicted_token = tokenizer.decode(predicted_id)
    print(f"Predicted: '{predicted_token}' (id={predicted_id})")
    print(f"Total: {total_ms:.1f}ms")

    # Build structured trace
    trace = tracer.to_json()

    # Print human-readable trace
    print(f"\n{'='*100}")
    print("INFERENCE TRACE — each row maps a Python call to a hardware unit")
    print(f"{'='*100}")
    print(f"{'#':>4}  {'HW Unit':<25} {'Module':<50} {'In':<25} {'Out':<25} {'ms':>8}")
    print(f"{'-'*100}")
    for e in trace:
        in_str = e["input"][0] if e["input"] else ""
        out_str = e["output"][0] if e["output"] else ""
        hw = e["hw_unit"] or "—"
        print(f"{e['step']:>4}  {hw:<25} {e['module']:<50} {in_str:<25} {out_str:<25} {e['ms']:>8.3f}")
    print(f"{'='*100}")

    # Write JSON trace
    with open(TRACE_OUT, "w") as f:
        json.dump({
            "model": "granite-4.0-h-350m",
            "prompt": prompt,
            "tokens": input_ids.tolist(),
            "predicted": predicted_token,
            "predicted_id": predicted_id,
            "total_ms": round(total_ms, 3),
            "trace": trace,
        }, f, indent=2)
    print(f"\nJSON trace written to {TRACE_OUT}")

    tracer.remove()


if __name__ == "__main__":
    main()
