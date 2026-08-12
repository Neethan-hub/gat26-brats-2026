#!/usr/bin/env python3
"""Synthetic CUDA preflight for the pinned nnU-Net ResEnc contract.

This harness imports nnU-Net's actual region Dice+BCE and deep-supervision
wrapper. It validates a patch-level forward/backward/checkpoint path and records
CUDA memory. It is intentionally not a full-case sliding-window, Docker, A10G
parity, or accuracy test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import tempfile
import time
from pathlib import Path


PINNED_NNUNET_VERSION = "2.8.1"
TOPOLOGY = {
    "input_channels": 4,
    "features_per_stage": (32, 64, 128, 256, 320, 320),
    "encoder_blocks": (1, 3, 4, 6, 6, 6),
    "decoder_convs": (1, 1, 1, 1, 1),
    "strides": (
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    ),
    "num_regions": 3,
}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_model(torch, deep_supervision: bool):
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    model = ResidualEncoderUNet(
        input_channels=TOPOLOGY["input_channels"],
        n_stages=len(TOPOLOGY["features_per_stage"]),
        features_per_stage=TOPOLOGY["features_per_stage"],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=((3, 3, 3),) * len(TOPOLOGY["features_per_stage"]),
        strides=TOPOLOGY["strides"],
        n_blocks_per_stage=TOPOLOGY["encoder_blocks"],
        num_classes=TOPOLOGY["num_regions"],
        n_conv_per_stage_decoder=TOPOLOGY["decoder_convs"],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    model.apply(model.initialize)
    return model


def build_native_region_loss(batch_dice: bool, outputs_count: int):
    import numpy as np
    from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

    base = DC_and_BCE_loss(
        {},
        {
            "batch_dice": batch_dice,
            "do_bg": True,
            "smooth": 1e-5,
            "ddp": False,
        },
        use_ignore_label=False,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    if outputs_count == 1:
        return base, [1.0]
    weights = np.asarray([1 / (2**i) for i in range(outputs_count)], dtype=float)
    # Mirrors nnUNetTrainer._build_loss for non-DDP execution in v2.8.1.
    weights[-1] = 0.0
    weights /= weights.sum()
    return DeepSupervisionWrapper(base, weights), weights.tolist()


def make_nested_target(torch, batch: int, patch: tuple[int, int, int], device):
    wt = torch.rand((batch, 1, *patch), device=device) < 0.08
    tc = wt & (torch.rand((batch, 1, *patch), device=device) < 0.55)
    et = tc & (torch.rand((batch, 1, *patch), device=device) < 0.35)
    return torch.cat((wt, tc, et), dim=1).float()


def tensor_tree(outputs):
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def targets_for_outputs(torch, target, outputs):
    import torch.nn.functional as functional

    result = []
    for output in tensor_tree(outputs):
        if tuple(output.shape[2:]) == tuple(target.shape[2:]):
            result.append(target)
        else:
            result.append(
                functional.interpolate(target, size=output.shape[2:], mode="nearest")
            )
    return result


def make_grad_scaler(torch):
    try:
        return torch.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def assert_finite_tensors(torch, tensors, label: str) -> None:
    for index, tensor in enumerate(tensors):
        if not bool(torch.isfinite(tensor).all().item()):
            raise SystemExit(f"FAIL: non-finite {label} tensor at index {index}")


def checkpoint_round_trip(torch, model, deep_supervision: bool) -> tuple[str, int]:
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        clone = build_model(torch, deep_supervision=deep_supervision)
        clone.load_state_dict(state, strict=True)
        clone_state = clone.state_dict()
        if state.keys() != clone_state.keys():
            raise SystemExit("FAIL: checkpoint key mismatch after strict reload")
        for key in state:
            if not torch.equal(state[key].cpu(), clone_state[key].cpu()):
                raise SystemExit(f"FAIL: checkpoint tensor mismatch for {key}")
        return digest, checkpoint.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", nargs=3, type=int, default=(160, 160, 128))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("train", "inference"), default="train")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--batch-dice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must match the generated nnU-Net plan when used as a pipeline check.",
    )
    parser.add_argument("--memory-limit-gib", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations cannot be negative")
    if args.memory_limit_gib is not None and (
        not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0
    ):
        parser.error("--memory-limit-gib must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        import torch
        import nnunetv2  # noqa: F401 - verifies importability
        import dynamic_network_architectures  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"FAIL: install the locked PyTorch/nnU-Net environment first: {exc}"
        ) from exc

    nnunet_version = package_version("nnunetv2")
    if nnunet_version != PINNED_NNUNET_VERSION:
        raise SystemExit(
            f"FAIL: expected nnunetv2=={PINNED_NNUNET_VERSION}, got {nnunet_version}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA GPU is required for this preflight")

    patch = tuple(args.patch)
    if len(patch) != 3 or any(value <= 0 or value % 32 for value in patch):
        raise SystemExit(
            "FAIL: fixed isotropic synthetic patch dimensions must be positive multiples of 32"
        )

    torch.manual_seed(21072026)
    torch.cuda.manual_seed_all(21072026)
    device = torch.device("cuda")
    deep_supervision = args.mode == "train"
    model = build_model(torch, deep_supervision=deep_supervision).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    architecture_repr_sha256 = hashlib.sha256(
        repr(model).encode("utf-8")
    ).hexdigest()

    x = torch.randn((args.batch_size, 4, *patch), device=device)
    optimizer = None
    scaler = None
    target = None
    native_loss = None
    loss_weights = []
    if args.mode == "train":
        target = make_nested_target(torch, args.batch_size, patch, device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-5,
        )
        scaler = make_grad_scaler(torch)

    def train_step(measure: bool) -> tuple[object, float | None, float]:
        assert optimizer is not None and scaler is not None and target is not None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if measure:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x)
            output_list = tensor_tree(outputs)
            assert_finite_tensors(torch, output_list, "output")
            nonlocal native_loss, loss_weights
            if native_loss is None:
                native_loss, loss_weights = build_native_region_loss(
                    args.batch_dice, len(output_list)
                )
            targets = targets_for_outputs(torch, target, output_list)
            loss_input = output_list if len(output_list) > 1 else output_list[0]
            target_input = targets if len(targets) > 1 else targets[0]
            loss = native_loss(loss_input, target_input)
        if not bool(torch.isfinite(loss).item()):
            raise SystemExit("FAIL: non-finite native region loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite_gradients = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise SystemExit("FAIL: non-finite gradient")
        scaler.step(optimizer)
        scaler.update()
        if measure:
            torch.cuda.synchronize()
        return outputs, float(loss.detach().cpu()), time.perf_counter() - start

    def inference_step(measure: bool) -> tuple[object, None, float]:
        model.eval()
        if measure:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            outputs = model(x)
        output_list = tensor_tree(outputs)
        assert_finite_tensors(torch, output_list, "output")
        if measure:
            torch.cuda.synchronize()
        return outputs, None, time.perf_counter() - start

    step = train_step if args.mode == "train" else inference_step
    outputs = None
    for _ in range(args.warmup_iterations):
        outputs, _, _ = step(False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    losses: list[float] = []
    for _ in range(args.iterations):
        outputs, loss_value, elapsed = step(True)
        times.append(elapsed)
        if loss_value is not None:
            losses.append(loss_value)

    assert outputs is not None
    output_shapes = [list(tensor.shape) for tensor in tensor_tree(outputs)]
    primary = output_shapes[0]
    if (
        primary[0] != args.batch_size
        or primary[1] != TOPOLOGY["num_regions"]
        or tuple(primary[2:]) != patch
    ):
        raise SystemExit(f"FAIL: bad primary output shape {primary}")

    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    if args.memory_limit_gib is not None and peak_reserved >= args.memory_limit_gib:
        raise SystemExit(
            f"FAIL: peak reserved {peak_reserved:.2f} GiB is not below "
            f"{args.memory_limit_gib:.2f} GiB"
        )

    checkpoint_hash, checkpoint_bytes = checkpoint_round_trip(
        torch, model, deep_supervision=deep_supervision
    )

    result = {
        "status": "PASS",
        "suite_version": "1.2",
        "scope": "synthetic_patch_only",
        "not_validated_by_this_test": [
            "segmentation accuracy",
            "protected GoAT data pipeline",
            "full-case sliding-window inference",
            "multi-fold ensemble memory/runtime",
            "Docker no-network execution",
            "A10G release parity unless the complete release pipeline is tested separately",
        ],
        "mode": args.mode,
        "patch": patch,
        "batch_size": args.batch_size,
        "batch_dice": args.batch_dice,
        "warmup_iterations": args.warmup_iterations,
        "measured_iterations": args.iterations,
        "topology": TOPOLOGY,
        "architecture_repr_sha256": architecture_repr_sha256,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "output_shapes": output_shapes,
        "native_loss": "nnunetv2.DC_and_BCE_loss(MemoryEfficientSoftDiceLoss)",
        "deep_supervision_weights": loss_weights,
        "amp": args.mode == "train",
        "dynamic_grad_scaler": args.mode == "train",
        "losses": losses,
        "seconds": times,
        "mean_seconds": sum(times) / len(times),
        "median_seconds": sorted(times)[len(times) // 2],
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "memory_limit_gib": args.memory_limit_gib,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint_bytes,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nnunetv2": nnunet_version,
        "dynamic_network_architectures": package_version(
            "dynamic-network-architectures"
        ),
        "numpy": package_version("numpy"),
        "python": platform.python_version(),
        "warning": (
            "PASS proves synthetic patch wiring/checkpoint/memory only. It is not "
            "an accuracy result or an A10G release certificate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
