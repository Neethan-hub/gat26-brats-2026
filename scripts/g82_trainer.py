#!/usr/bin/env python3
"""GAT-26 Stage G82 — repository-local nnU-Net trainer and data-loader subclasses.

Nothing in site-packages is patched. Everything here subclasses the pinned
nnU-Net 2.8.1 classes actually installed in the production environment.

Candidates (frozen at preregistration, may not be extended):

  C0   the immutable baseline policy (no G82 training)
  T    label-tail-aware subject sampling; native augmentation, native loss
  DG   native sampling plus exactly one low-frequency MRI appearance transform
  TDG  T plus DG

Configuration arrives through the ``G82_CONFIG`` environment variable pointing at
a JSON file, so no credential or private path ever appears on a command line.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple, Union

import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# --------------------------------------------------------------------------- #
# frozen constants
# --------------------------------------------------------------------------- #

RECIPES = ("C0", "T", "DG", "TDG")          # closed set; see tests/test_g82_science.py

C0_INITIAL_LR = 1e-2                        # nnUNetTrainer default, i.e. what C0 used
FINETUNE_LR_FRACTION = 0.05                 # 5 % of the original initial learning rate
FINETUNE_LR_CAP = 5e-4
FINETUNE_LR = min(C0_INITIAL_LR * FINETUNE_LR_FRACTION, FINETUNE_LR_CAP)

# subject-level tail-aware mixture (must sum to 1.0)
T_MIXTURE = {
    "uniform": 0.50,
    "small_et": 0.15,
    "small_tc": 0.10,
    "multifocal": 0.075,
    "et_absent": 0.10,
    "wt_extreme": 0.075,
}
T_CAP_MULTIPLE = 3.0                        # cap per-case probability at 3x uniform

# region choice *within* an already-requested foreground patch
T_REGION_WEIGHTS = {"ET": 0.50, "TC": 0.30, "WT": 0.20}
REGION_PARENT = {"ET": "TC", "TC": "WT", "WT": None}

# the single DG appearance transform
DG_P = 0.20
DG_GRID = (4, 4, 4)
DG_A_RANGE = (-0.20, 0.20)
DG_B_STD_FRACTION = 0.10

REGION_DEFS = {"WT": frozenset((1, 2, 3)), "TC": frozenset((1, 3)), "ET": frozenset((3,))}

_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def strip_wrappers(key: str) -> str:
    """Remove DDP / torch.compile state-dict prefixes, in any order."""
    changed = True
    while changed:
        changed = False
        for p in _WRAPPER_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


# --------------------------------------------------------------------------- #
# DG: one label-preserving low-frequency per-modality appearance field
# --------------------------------------------------------------------------- #

class G82AppearanceTransform(ImageOnlyTransform):
    """x' = x * (1 + a*f) + b*f, applied only inside the nonzero foreground.

    ``f`` is an independent zero-mean 4x4x4 smooth field per modality, trilinearly
    upsampled to the patch and rescaled so that max|f| == 1. Labels, geometry,
    dtype, shape and background zeros are untouched by construction.
    """

    def __init__(self, grid=DG_GRID, a_range=DG_A_RANGE, b_std_fraction=DG_B_STD_FRACTION):
        super().__init__()
        self.grid = tuple(grid)
        self.a_range = tuple(a_range)
        self.b_std_fraction = float(b_std_fraction)

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _field(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        g = torch.rand(self.grid, dtype=torch.float32) * 2.0 - 1.0
        g = g - g.mean()                                     # zero-mean by construction
        f = torch.nn.functional.interpolate(
            g[None, None], size=tuple(shape), mode="trilinear", align_corners=True)[0, 0]
        m = f.abs().max()
        if m > 0:
            f = f / m                                        # max|f| == 1
        return f

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if img.ndim != 4:                                    # (C, X, Y, Z) expected
            return img
        out = img.clone()
        for c in range(out.shape[0]):
            x = out[c]
            fg = x != 0
            if not bool(fg.any()):
                continue                                     # all-zero channel: leave as is
            fg_std = float(x[fg].std())
            f = self._field(tuple(x.shape)).to(x.dtype)
            a = float(np.random.uniform(*self.a_range))
            b = float(np.random.uniform(-self.b_std_fraction * fg_std,
                                        self.b_std_fraction * fg_std))
            new = x * (1.0 + a * f) + b * f
            x[fg] = new[fg]                                  # background zeros preserved
        return out


def insert_dg_transform(composed: ComposeTransforms) -> ComposeTransforms:
    """Append the DG transform to the *image* stage of a native nnU-Net pipeline.

    It is inserted immediately before the first segmentation-shaping transform
    (region conversion / deep-supervision downsampling) so every native intensity
    and spatial augmentation is retained exactly and none is duplicated.
    """
    tail_names = ("ConvertSegmentationToRegionsTransform", "DownsampleSegForDSTransform")
    ts = list(composed.transforms)
    cut = len(ts)
    for i, t in enumerate(ts):
        if type(t).__name__ in tail_names:
            cut = i
            break
    ts.insert(cut, RandomTransform(G82AppearanceTransform(), apply_probability=DG_P))
    return ComposeTransforms(ts)


# --------------------------------------------------------------------------- #
# T: tail-aware sampling
# --------------------------------------------------------------------------- #

def region_of_key(key) -> Union[str, None]:
    """Map an nnU-Net class_locations key to ET / TC / WT, or None."""
    if isinstance(key, (tuple, list, set, frozenset)):
        s = frozenset(int(x) for x in key)
    else:
        try:
            s = frozenset((int(key),))
        except (TypeError, ValueError):
            return None
    for name, definition in REGION_DEFS.items():
        if s == definition:
            return name
    return None


def region_weights_for(eligible: List[str]) -> List[float]:
    """T's ET/TC/WT weights restricted to the regions actually present.

    Mass belonging to an absent region flows to its nearest available parent
    (ET -> TC -> WT). If no parent is available the mass is spread uniformly.
    """
    w = {r: 0.0 for r in eligible}
    for r, m in T_REGION_WEIGHTS.items():
        target = r
        while target is not None and target not in w:
            target = REGION_PARENT[target]
        if target is None:
            for k in w:
                w[k] += m / len(w)
        else:
            w[target] += m
    tot = sum(w.values())
    return [w[r] / tot for r in eligible]


def tail_sampling_probabilities(strata: dict, identifiers: List[str]) -> np.ndarray:
    """Subject-level probabilities from the preregistered mixture.

    ``strata`` maps each identifier to the list of stratum names it belongs to.
    Empty strata donate their probability to uniform sampling. Every case is then
    capped at ``T_CAP_MULTIPLE`` times the uniform probability and renormalised.
    """
    n = len(identifiers)
    assert n > 0
    members = {s: [i for i in identifiers if s in strata.get(i, ())] for s in T_MIXTURE
               if s != "uniform"}
    uniform_mass = T_MIXTURE["uniform"]
    for s, m in members.items():
        if not m:
            uniform_mass += T_MIXTURE[s]                     # redistribute empty stratum

    p = np.full(n, uniform_mass / n, dtype=np.float64)
    index = {ident: k for k, ident in enumerate(identifiers)}
    for s, m in members.items():
        if not m:
            continue
        share = T_MIXTURE[s] / len(m)
        for ident in m:
            p[index[ident]] += share

    cap = T_CAP_MULTIPLE / n
    for _ in range(64):                                      # capped water-filling
        over = p > cap
        if not over.any():
            break
        excess = float((p[over] - cap).sum())
        p[over] = cap
        room = ~over
        if not room.any():
            break
        p[room] += excess / int(room.sum())
    p = p / p.sum()
    return p


class G82TailDataLoader(nnUNetDataLoader):
    """nnUNetDataLoader with T's region preference inside foreground patches.

    Subject-level tail weighting is applied through the native
    ``sampling_probabilities`` hook; only the region choice needs an override.
    C0's total foreground-oversampling fraction is untouched.
    """

    use_region_weights = True

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None,
                 verbose=False, **kwargs):
        if force_fg and class_locations and overwrite_class is None and self.use_region_weights:
            eligible = [k for k in class_locations
                        if len(class_locations[k]) > 0 and region_of_key(k) is not None]
            if eligible:
                names = [region_of_key(k) for k in eligible]
                probs = region_weights_for(names)
                overwrite_class = eligible[int(np.random.choice(len(eligible), p=probs))]
        return super().get_bbox(data_shape, force_fg, class_locations,
                                overwrite_class=overwrite_class, verbose=verbose, **kwargs)


# --------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------- #

def load_g82_config() -> dict:
    path = os.environ.get("G82_CONFIG")
    if not path:
        return {"recipe": "C0", "epochs": 40, "seed": 20260730}
    with open(path) as f:
        cfg = json.load(f)
    assert cfg["recipe"] in RECIPES, f"recipe {cfg['recipe']!r} is not in the frozen set"
    return cfg


class nnUNetTrainerG82(nnUNetTrainer):
    """Fine-tunes from copied original C0 weights.

    Same optimizer family, momentum/Nesterov, weight decay, batch size, patch size,
    deep supervision, Dice+BCE loss, plans and architecture as C0. Only the maximum
    learning rate, the epoch budget and (per recipe) the sampler/augmentation differ.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        # nnU-Net 2.8.1 pops this key out of the plans dict; copy so the caller's
        # plans object is never mutated.
        plans = dict(plans)
        plans.setdefault("continue_training", False)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        cfg = load_g82_config()
        self.g82_recipe = cfg["recipe"]
        self.g82_init_checkpoint = cfg.get("init_checkpoint")
        self.g82_strata_file = cfg.get("strata_file")
        self.g82_seed = int(cfg.get("seed", 20260730))
        self.num_epochs = int(cfg["epochs"])
        self.initial_lr = FINETUNE_LR                       # 5e-4, poly-decayed over num_epochs
        self.g82_init_report = None
        # preregistered discovery checkpoints; saved atomically under their own names
        self.g82_checkpoint_epochs = sorted(int(e) for e in cfg.get("checkpoint_epochs", []))

    # -- initialisation from copied C0 weights ------------------------------ #

    def initialize(self):
        super().initialize()
        if self.g82_init_checkpoint:
            self.g82_init_report = self.load_c0_weights(self.g82_init_checkpoint)

    def load_c0_weights(self, path: str) -> dict:
        """Load every compatible parameter, including the segmentation heads.

        Fails loudly rather than silently dropping tensors: nnU-Net's own
        ``load_checkpoint`` tolerates key mismatches, which would quietly discard
        segmentation-head weights and invalidate the C0-equivalence gate.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sd = ckpt["network_weights"] if "network_weights" in ckpt else ckpt
        sd = {strip_wrappers(k): v for k, v in sd.items()}
        net = self.network.module if self.is_ddp else self.network
        net = getattr(net, "_orig_mod", net)          # unwrap torch.compile
        missing, unexpected = net.load_state_dict(sd, strict=True)
        report = {
            "loaded_tensors": len(sd),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "seg_head_tensors": sorted(k for k in sd if "seg_layers" in k),
        }
        assert not report["missing_keys"] and not report["unexpected_keys"]
        assert report["seg_head_tensors"], "segmentation-head weights were not loaded"
        return report

    # -- preregistered intermediate checkpoints ----------------------------- #

    def on_epoch_end(self):
        super().on_epoch_end()
        # nnUNetTrainer.on_epoch_end increments current_epoch, so after the super()
        # call it already equals the number of completed epochs.
        done = self.current_epoch
        if done in self.g82_checkpoint_epochs:
            path = os.path.join(self.output_folder, f"checkpoint_epoch{done}.pth")
            tmp = path + ".tmp"
            self.save_checkpoint(tmp)
            os.replace(tmp, path)                           # atomic promotion
            self.print_to_log_file(f"G82: saved preregistered checkpoint at epoch {done}")

    # -- T: subject-level sampling ------------------------------------------ #

    def g82_sampling_probabilities(self, identifiers: List[str]):
        if self.g82_recipe not in ("T", "TDG"):
            return None
        with open(self.g82_strata_file) as f:
            strata = json.load(f)
        strata = {k: tuple(v) for k, v in strata.items()}
        missing = [i for i in identifiers if i not in strata]
        assert not missing, f"{len(missing)} training identifiers have no stratum record"
        return tail_sampling_probabilities(strata, list(identifiers))

    # -- DG: augmentation ---------------------------------------------------- #

    def get_training_transforms(self, *args, **kwargs):
        base = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        if self.g82_recipe in ("DG", "TDG"):
            return insert_dg_transform(base)
        return base

    # -- wire the loaders ---------------------------------------------------- #

    def get_dataloaders(self):
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import \
            NonDetMultiThreadedAugmenter

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        ds_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        regions = self.label_manager.foreground_regions if self.label_manager.has_regions else None
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, ds_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=regions, ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            ds_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=regions, ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        probs = self.g82_sampling_probabilities(list(dataset_tr.identifiers))
        loader_cls = G82TailDataLoader if self.g82_recipe in ("T", "TDG") else nnUNetDataLoader

        dl_tr = loader_cls(dataset_tr, self.batch_size, initial_patch_size,
                           self.configuration_manager.patch_size, self.label_manager,
                           oversample_foreground_percent=self.oversample_foreground_percent,
                           sampling_probabilities=probs, pad_sides=None,
                           transforms=tr_transforms,
                           probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None,
                                  transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        n_proc = get_allowed_n_proc_DA()
        if n_proc == 0:
            mt_train = SingleThreadedAugmenter(dl_tr, None)
            mt_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=n_proc,
                num_cached=max(6, n_proc // 2), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
            mt_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, n_proc // 2),
                num_cached=max(3, n_proc // 4), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
        _ = next(mt_train)
        _ = next(mt_val)
        return mt_train, mt_val
