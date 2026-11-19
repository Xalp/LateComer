"""export_ckpt.py -- make an evaluation copy of a late-entry training checkpoint (weights + config + tokenizer only, no optimizer states).
usage: python export_ckpt.py <run>/checkpoint-<N> <dest dir>        (refuses an existing <dest>/.done; writes .done last)
       python export_ckpt.py --fix-config-only <dir>                 (config rewrite in place; used by launch_late.sh's publish step)

Why: lmeval_ouro/run_unit.sh stages EVERYTHING in the model dir (a training checkpoint also holds ~35 GB of DeepSpeed states), and
config.json of a training checkpoint names the training class. The exported config is exactly what ouro_stream.load / generate.py expect:
model_type a2d-ouro, architectures [A2DOuroLMHeadModel] (cosmetic: load() passes the class explicitly), no auto_map; weights unchanged
(same parameter names as every A2D-Ouro checkpoint). Files are copied, then sizes verified; nothing in <src> is touched.
"""
import json, os, shutil, sys

KEEP_EXT = (".safetensors", ".json", ".txt", ".jinja")
SKIP = {"trainer_state.json", "late_manifest.json"}


def fix_config(d):
    p = os.path.join(d, "config.json"); c = json.load(open(p))
    c.pop("auto_map", None); c["model_type"] = "a2d-ouro"; c["architectures"] = ["A2DOuroLMHeadModel"]
    json.dump(c, open(p + ".tmp", "w"), indent=2); os.replace(p + ".tmp", p)
    return c


def main():
    if sys.argv[1] == "--fix-config-only":
        c = fix_config(sys.argv[2]); print(f"[export] config fixed in {sys.argv[2]}: {c['model_type']} {c['architectures']}"); return
    src, dst = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    if os.path.exists(os.path.join(dst, ".done")): print(f"[export] {dst}/.done exists: not overwriting"); return
    files = [f for f in sorted(os.listdir(src)) if f.endswith(KEEP_EXT) and f not in SKIP and not f.startswith("late_manifest") and os.path.isfile(os.path.join(src, f))]
    assert any(f.endswith(".safetensors") for f in files), f"no weights in {src}"
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"): assert f in files, f"{src} lacks {f}"
    os.makedirs(dst, exist_ok=True)
    for f in files:
        s = os.path.getsize(os.path.join(src, f)); t = os.path.join(dst, f)
        if not (os.path.exists(t) and os.path.getsize(t) == s): shutil.copyfile(os.path.join(src, f), t)
        assert os.path.getsize(t) == s, f"size mismatch {f}"
    c = fix_config(dst)
    json.dump(dict(src=src, files=files, total_ut_steps=c.get("total_ut_steps"), vocab_size=c.get("vocab_size")), open(os.path.join(dst, "export.json"), "w"), indent=1)
    open(os.path.join(dst, ".done"), "w").close()
    print(f"[export] {src} -> {dst}: {len(files)} files, {sum(os.path.getsize(os.path.join(dst, f)) for f in files) / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
