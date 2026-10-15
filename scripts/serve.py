"""Serve a FoveaVLA checkpoint over WebSocket."""

import argparse
import logging
from pathlib import Path

from foveavla_client import foveation

from foveavla.policies import policy_config
from foveavla.serving import websocket_policy_server
from foveavla.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("foveavla_single", "foveavla_dual"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to checkpoints/<model>; override for trained weights")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fast-foundation-repository", type=Path, default=Path(".runtime/Fast-FoundationStereo"))
    parser.add_argument("--fast-foundation-checkpoint", type=Path)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or Path("checkpoints") / args.model
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / "assets/droid/norm_stats.json").is_file():
        parser.error("checkpoint must contain params/ and assets/droid/norm_stats.json")
    cfg = config.get_config(args.model)
    metadata = foveation.with_legacy_fields(cfg.policy_metadata)
    from foveavla.shared.fast_foundation_runtime import FastFoundationStereoRuntime

    weights = args.fast_foundation_checkpoint or (
        args.fast_foundation_repository / "weights/23-36-37/model_best_bp2_serialize.pth"
    )
    runtime = FastFoundationStereoRuntime(
        repository=args.fast_foundation_repository,
        checkpoint=weights,
        repository_commit=metadata["fast_foundation_repository_commit"],
        checkpoint_sha256=metadata["fast_foundation_checkpoint_sha256"],
        input_resolution=tuple(metadata["fast_foundation_input_resolution"]),
        max_disparity=int(metadata["fast_foundation_max_disparity"]),
        valid_iters=int(metadata["fast_foundation_valid_iters"]),
        low_memory=bool(metadata["fast_foundation_low_memory"]),
    )
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    from foveavla.policies.fast_foundation_depth_policy import FastFoundationDepthPolicy

    policy = FastFoundationDepthPolicy(policy, runtime, metadata)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
