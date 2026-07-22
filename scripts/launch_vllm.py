import argparse
import json
import os
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path

_QWEN3_OMNI_THINKER_DEPLOY = "qwen3_omni_moe_thinker_only.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for hidden states extraction",
        usage=(
            "launch_vllm.py [-h] MODEL [--hidden-states-path HIDDEN_STATES_PATH] "
            "[--target-layer-ids TARGET_LAYER_IDS [TARGET_LAYER_IDS ...]] -- *VLLM_ARGS"
        ),
    )
    parser.add_argument(
        "model", type=str, help="Model name or path to extract hidden states from"
    )
    parser.add_argument(
        "--hidden-states-path",
        type=str,
        default="/tmp/hidden_states",  # noqa: S108
        help="The directory to save hidden states to. Default '/tmp/hidden_states'.",
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of integer layer ids. Defaults to "
            "[2, num_hidden_layers // 2, num_hidden_layers - 3]. "
            "Note: if set, you must also pass the same value into the training process"
        ),
    )
    parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last layer (num_hidden_layers) to "
            "target_layer_ids for verifier hidden states extraction. Default: True"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    parser.add_argument(
        "--omni",
        action="store_true",
        help=(
            "Launch the vLLM-Omni Thinker-only server and apply extraction "
            "settings to stage 0."
        ),
    )
    parser.add_argument(
        "--omni-deploy-config",
        type=Path,
        default=None,
        help=(
            "Thinker-only vLLM-Omni deploy YAML. By default it is resolved "
            "from the installed vllm_omni package."
        ),
    )
    return parser.parse_known_args()


def resolve_omni_deploy_config(config_path: Path | None) -> Path:
    if config_path is not None:
        resolved = config_path.expanduser().resolve()
    else:
        spec = find_spec("vllm_omni")
        package_dirs = spec.submodule_search_locations if spec is not None else None
        if not package_dirs:
            raise RuntimeError(
                "Cannot find vllm_omni. Run this script with the vLLM-Omni "
                "environment or pass --omni-deploy-config."
            )
        resolved = (
            Path(next(iter(package_dirs))) / "deploy" / _QWEN3_OMNI_THINKER_DEPLOY
        )

    if not resolved.is_file():
        raise FileNotFoundError(f"vLLM-Omni deploy config not found: {resolved}")
    return resolved


def main():
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")

    from transformers import AutoConfig  # noqa: PLC0415

    from speculators.models.utils import (  # noqa: PLC0415
        resolve_verifier_text_config,
    )

    config = resolve_verifier_text_config(AutoConfig.from_pretrained(args.model))
    num_hidden_layers = config.num_hidden_layers

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly passed into the training script.",
            stacklevel=2,
        )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": args.hidden_states_path},
    }

    if args.omni:
        deploy_config = resolve_omni_deploy_config(args.omni_deploy_config)
        stage_overrides = {
            "0": {
                "speculative_config": speculative_config,
                "kv_transfer_config": kv_transfer_config,
            }
        }
        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            args.model,
            "--omni",
            "--deploy-config",
            str(deploy_config),
            "--stage-overrides",
            json.dumps(stage_overrides),
            *vllm_args,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--speculative_config",
            json.dumps(speculative_config),
            "--kv_transfer_config",
            json.dumps(kv_transfer_config),
            *vllm_args,
        ]

        disable_cp_arg = "--no-enable-chunked-prefill"
        if disable_cp_arg not in cmd:
            cmd.append(disable_cp_arg)

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
