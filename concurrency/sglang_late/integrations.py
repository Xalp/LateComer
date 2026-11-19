"""integrations.py -- process-level setup: config parser + architecture override, model registration, pool patch.

The config-parser pattern is COPIED from LoopSpec (baselines/loopspec/sglang_recurrent/integrations/sglang.py,
install_sglang_config_compatibility): wrap SGLang's built-in "hf" parser and rewrite `architectures` for Ouro configs so ModelRunner picks
our serving class. Serving v2 uses LoopSpec's recipe (loopspec): the stock Ouro config.json (model_type ouro,
configuration_ouro.py) with vocab_size = 49216 and pad_token_id = 0, next to v2's weights and tokenizer.
"""
import json, logging, os

logger = logging.getLogger(__name__)
PARSER_NAME = "ouro_late"
ARCH = "OuroLateForCausalLM"
_installed = False


def install_config_parser():
    global _installed
    if _installed: return PARSER_NAME
    from sglang.srt.configs.model_config_parser_registry import ModelConfigParserBase, get_model_config_parser, register_model_config_parser
    from sglang.srt.utils.hf_transformers import config as _config  # noqa: F401  (registers the built-in parsers)
    hf_parser = get_model_config_parser("hf")

    @register_model_config_parser(PARSER_NAME)
    class OuroLateConfigParser(ModelConfigParserBase):
        def parse(self, model, trust_remote_code, revision=None, **kwargs):
            config = hf_parser.parse(model, trust_remote_code=trust_remote_code, revision=revision, **kwargs)
            if getattr(config, "model_type", None) in ("ouro", "a2d-ouro"):
                config.architectures = [ARCH]
            return config

    _installed = True
    return PARSER_NAME


def register_models():
    from sglang.srt.models.registry import ModelRegistry
    ModelRegistry.register("sglang_late.models", overwrite=True, strict=True)


def child_setup(server_args):
    """run in every SGLang child process (scheduler / detokenizer) before anything else of ours is used"""
    from .spec_worker import register, late_settings
    from .kv_pool import install_pool_patch
    register()
    install_config_parser()
    register_models()
    cfgj = json.load(open(os.path.join(server_args.model_path, "config.json")))
    s = late_settings()
    install_pool_patch(cfgj.get("total_ut_steps", 4), cfgj["num_hidden_layers"], max(1, s["Rmax"] * s["PH"]))
