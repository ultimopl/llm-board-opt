from __future__ import annotations

import struct
from typing import Any

from board_opt.gguf_io import TYPE_NAMES, read_gguf_header, tensor_nbytes
from board_opt.models import GGUFInfo, HardwareProfile
from board_opt.optimizer import plan_moe_expert_steps, plan_requant_targets
from board_opt.quant_bench import QuantBenchEntry
from board_opt.vocab_prune import SINGLE_BYTE_TOKENS, U2B


def _tensor_role(name: str) -> str:
    if name == "token_embd.weight":
        return "token_embd"
    if name == "output.weight":
        return "output"
    if "norm" in name or "bias" in name:
        return "norm_bias"
    if "loop" in name and "ctrl" in name:
        return "loop_ctrl"
    if "ffn_gate_inp" in name or "exp_probs" in name:
        return "moe_router"
    if "_exps." in name:
        return "moe_experts"
    if "attn_" in name:
        return "attn"
    if "ffn_" in name:
        return "ffn"
    return "other"


def summarize_layer_quants(
    info: GGUFInfo,
    hw: HardwareProfile,
    quant_table: dict[str, QuantBenchEntry] | None = None,
    vram_constrained: bool = False,
) -> list[dict[str, Any]]:
    targets = plan_requant_targets(info.tensors, hw, quant_table=quant_table, vram_constrained=vram_constrained)
    groups: dict[tuple[str, int, int | None], dict[str, Any]] = {}

    for t in info.tensors:
        role = _tensor_role(t.name)
        dst_id = targets.get(t.name)
        key = (role, t.ttype, dst_id)
        if key not in groups:
            groups[key] = {"role": role, "src_id": t.ttype, "dst_id": dst_id, "count": 0, "src_bytes": 0, "dst_bytes": 0}
        g = groups[key]
        g["count"] += 1
        g["src_bytes"] += t.nbytes
        g["dst_bytes"] += tensor_nbytes(t.dims, dst_id) if dst_id is not None else t.nbytes

    rows: list[dict[str, Any]] = []
    for g in sorted(groups.values(), key=lambda x: -x["src_bytes"]):
        src_name = TYPE_NAMES.get(g["src_id"], str(g["src_id"]))
        dst_name = TYPE_NAMES.get(g["dst_id"], "-") if g["dst_id"] is not None else "-"
        save_mb = (g["src_bytes"] - g["dst_bytes"]) / (1024.0 * 1024.0)
        tg_gain_pct = 0.0
        target_cos = None
        if quant_table and dst_name != "-":
            se = quant_table.get(src_name.upper())
            de = quant_table.get(dst_name.upper())
            if se and de and se.tg_gws > 0:
                tg_gain_pct = ((de.tg_gws - se.tg_gws) / se.tg_gws) * 100.0
            if de:
                target_cos = de.cos_sim
        rows.append({
            "role": g["role"],
            "count": g["count"],
            "src_type": src_name,
            "size_mb": round(g["src_bytes"] / (1024.0 * 1024.0), 1),
            "target_type": dst_name,
            "est_vram_save_mb": round(save_mb, 1),
            "est_tg_gain_pct": round(tg_gain_pct, 1),
            "target_cos_sim": target_cos,
        })
    return rows


def summarize_moe_experts(info: GGUFInfo, vram_constrained: bool = False) -> dict[str, Any] | None:
    if info.expert_count <= 0:
        return None
    exp_bytes = sum(t.nbytes for t in info.tensors if "_exps." in t.name)
    exp_mb = exp_bytes / (1024.0 * 1024.0)
    plan = plan_moe_expert_steps(info, vram_constrained=vram_constrained)
    options: list[dict[str, Any]] = []
    for k_act in plan["active_candidates"]:
        ratio = info.expert_used_count / max(1, k_act)
        options.append({
            "kind": "active_experts",
            "change": f"{info.expert_used_count} -> {k_act}",
            "est_vram_save_mb": 0.0,
            "est_tg_gain_pct": round((ratio - 1.0) * 45.0, 1),  # MoE FFN is ~45-55% of decode traffic
            "risk": "low" if k_act >= 6 else "medium (may drop reasoning/code)",
        })
    for frac in (0.75, 0.50):
        tgt = max(info.expert_used_count * 2, int(info.expert_count * frac))
        if tgt < info.expert_count:
            options.append({
                "kind": "reap_total_experts",
                "change": f"{info.expert_count} -> {tgt}",
                "est_vram_save_mb": round(exp_mb * (1.0 - tgt / info.expert_count), 1),
                "est_tg_gain_pct": 0.0,
                "risk": "medium (pp +6-12%, saves VRAM, raises PPL)",
            })
    return {
        "expert_count": info.expert_count,
        "expert_used_count": info.expert_used_count,
        "experts_mb": round(exp_mb, 1),
        "options": options,
    }


def _classify_script(tok: str, is_special: bool) -> str:
    if is_special or tok in SINGLE_BYTE_TOKENS:
        return "special_byte"
    try:
        raw = bytes(U2B[ch] for ch in tok)
        s = raw.decode("utf-8")
    except Exception:
        return "partial_utf8"
    has_cjk = has_cyr = has_arab = has_latin_ext = has_other = False
    for ch in s:
        cp = ord(ch)
        if cp < 0x80:
            continue
        if 0x00A0 <= cp <= 0x017F:
            has_latin_ext = True
        elif (0x4E00 <= cp <= 0x9FFF) or (0x3040 <= cp <= 0x30FF) or (0xAC00 <= cp <= 0xD7AF):
            has_cjk = True
        elif 0x0400 <= cp <= 0x04FF:
            has_cyr = True
        elif 0x0600 <= cp <= 0x06FF:
            has_arab = True
        else:
            has_other = True
    if has_cjk:
        return "cjk"
    if has_cyr:
        return "cyrillic"
    if has_arab:
        return "arabic"
    if has_other:
        return "other_unicode"
    if has_latin_ext:
        return "latin_ext"
    return "ascii_en_code"


def summarize_vocab_scripts(gguf_path: str) -> dict[str, Any]:
    _, kvs, tensors, _, _ = read_gguf_header(gguf_path)
    kv_map = {k: (vt, val) for k, vt, val in kvs}
    tok_model = str(kv_map.get("tokenizer.ggml.model", (8, "unknown"))[1])
    if "tokenizer.ggml.tokens" not in kv_map:
        return {"tokenizer_model": tok_model, "prunable": False, "vocab_size": 0, "emb_mb": 0.0, "scripts": {}}

    tokens: list[str] = list(kv_map["tokenizer.ggml.tokens"][1][1])
    n_vocab = len(tokens)
    token_types: list[int] | None = None
    if "tokenizer.ggml.token_type" in kv_map:
        _, raw_tt, cnt = kv_map["tokenizer.ggml.token_type"][1]
        if isinstance(raw_tt, (bytes, bytearray)):
            token_types = list(struct.unpack(f"<{cnt}i", raw_tt))

    emb_bytes = sum(
        tensor_nbytes(t["dims"], t["type"])
        for t in tensors
        if t["name"] in ("token_embd.weight", "output.weight")
    )
    bytes_per_tok = emb_bytes / max(1, n_vocab)

    counts: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        is_spec = token_types is not None and token_types[i] != 1
        cat = _classify_script(tok, is_spec)
        counts[cat] = counts.get(cat, 0) + 1

    scripts = {
        cat: {
            "count": cnt,
            "pct": round(100.0 * cnt / max(1, n_vocab), 1),
            "est_vram_save_mb": round((cnt * bytes_per_tok) / (1024.0 * 1024.0), 1),
        }
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1])
    }
    return {
        "tokenizer_model": tok_model,
        "prunable": tok_model == "gpt2",
        "vocab_size": n_vocab,
        "emb_mb": round(emb_bytes / (1024.0 * 1024.0), 1),
        "scripts": scripts,
    }


def format_inspection_report(
    layer_rows: list[dict[str, Any]],
    moe: dict[str, Any] | None,
    vocab: dict[str, Any] | None,
) -> str:
    out: list[str] = []

    out.append("=== 1. Layer Quants & Recommended Board-Native Cuts ===")
    out.append(f"{'Role':<12} | {'Cnt':>4} | {'Quant':<8} | {'Size MiB':>8} | {'Target':<8} | {'Save MiB':>8} | {'Est tg':>7} | {'CosSim':>8}")
    out.append("-" * 82)
    for r in layer_rows:
        cos_s = f"{r['target_cos_sim']:.5f}" if r["target_cos_sim"] is not None else "-"
        tg_s = f"{r['est_tg_gain_pct']:+.1f}%" if r["target_type"] != "-" else "-"
        out.append(
            f"{r['role']:<12} | {r['count']:4d} | {r['src_type']:<8} | {r['size_mb']:8.1f} | "
            f"{r['target_type']:<8} | {r['est_vram_save_mb']:8.1f} | {tg_s:>7} | {cos_s:>8}"
        )

    if moe:
        out.append("")
        out.append(f"=== 2. Prunable MoE Experts ({moe['expert_count']} total, {moe['expert_used_count']} active, {moe['experts_mb']:.1f} MiB) ===")
        out.append(f"{'Kind':<20} | {'Change':<12} | {'Save MiB':>8} | {'Est tg':>7} | {'Expected Loss / Risk'}")
        out.append("-" * 82)
        for o in moe["options"]:
            tg_s = f"{o['est_tg_gain_pct']:+.1f}%" if o["est_tg_gain_pct"] > 0 else "-"
            out.append(f"{o['kind']:<20} | {o['change']:<12} | {o['est_vram_save_mb']:8.1f} | {tg_s:>7} | {o['risk']}")

    if vocab:
        out.append("")
        status = "prunable (GPT-2 BPE)" if vocab["prunable"] else f"read-only ({vocab['tokenizer_model']})"
        out.append(f"=== 3. Vocabulary Scripts ({vocab['vocab_size']} tokens, {vocab['emb_mb']:.1f} MiB emb+lm_head, {status}) ===")
        out.append(f"{'Script / Group':<18} | {'Tokens':>8} | {'Share':>6} | {'Save MiB if cut':>15}")
        out.append("-" * 58)
        for cat, s in vocab["scripts"].items():
            out.append(f"{cat:<18} | {s['count']:8d} | {s['pct']:5.1f}% | {s['est_vram_save_mb']:15.1f}")

    return "\n".join(out)
