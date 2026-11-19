"""Lightweight lm-evaluation-harness adapters (DumpLM & ReplayLM) for zero-shot chat-template evaluation."""
import json

MODELS = {
    "v2": "Xalphinions/ouro-2.6b-late-entry",
    "early": "Xalphinions/ouro-2.6b-early-exit",
    "stock26": "ByteDance/Ouro-2.6B",
    "stock14": "ByteDance/Ouro-1.4B",
}


def make_lm_classes():
    from lm_eval.api.model import LM
    from transformers import AutoTokenizer

    class OuroLM(LM):
        def __init__(self, tokenizer_path: str):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            self._rank, self._world_size = 0, 1

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                enable_thinking=False,
            )

        @property
        def tokenizer_name(self):
            return self.tokenizer.name_or_path.replace("/", "__")

        @property
        def rank(self):
            return self._rank

        @property
        def world_size(self):
            return self._world_size

        @property
        def device(self):
            return "cpu"

        def loglikelihood(self, requests):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

    class DumpLM(OuroLM):
        def __init__(self, tokenizer_path: str):
            super().__init__(tokenizer_path)
            self.records = []

        def generate_until(self, requests, disable_tqdm=False):
            out = []
            for req in requests:
                context, gk = req.args[0], dict(req.args[1] or {})
                until = gk.get("until", []) or []
                if isinstance(until, str):
                    until = [until]
                self.records.append(
                    dict(
                        task=req.task_name,
                        doc_id=req.doc_id,
                        context=context,
                        until=list(until),
                        gen_kwargs={k: v for k, v in gk.items() if k != "until"},
                    )
                )
                out.append("")
            return out

    class ReplayLM(OuroLM):
        def __init__(self, tokenizer_path: str, gens: dict, contexts: dict = None):
            super().__init__(tokenizer_path)
            self.gens = gens
            self.contexts = contexts or {}
            self.missing = []
            self.mismatch = []

        def generate_until(self, requests, disable_tqdm=False):
            out = []
            for req in requests:
                key = (req.task_name, req.doc_id)
                ctx = req.args[0]
                if key in self.contexts and self.contexts[key] != ctx:
                    self.mismatch.append(key)
                if key not in self.gens:
                    self.missing.append(key)
                    out.append("")
                else:
                    out.append(self.gens[key])
            return out

    return OuroLM, DumpLM, ReplayLM


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
