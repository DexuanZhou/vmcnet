"""Host-side, threshold-triggered snapshots for rare VMC tail bursts.

The feature is deliberately opt-in.  When disabled, :func:`maybe_save_burst`
returns before transferring any device arrays or touching the filesystem.
"""

import json
import os
from typing import Any, Mapping, MutableMapping, Optional

import jax
import numpy as np

from vmcnet.utils import io


_PAYLOAD_PREFIX = "burst_diag_"


def pop_payload(metrics: MutableMapping[str, Any]) -> dict:
    """Remove large diagnostic payloads so normal metric writers never see them."""
    keys = [key for key in metrics if key.startswith(_PAYLOAD_PREFIX)]
    return {key[len(_PAYLOAD_PREFIX):]: metrics.pop(key) for key in keys}


def should_capture(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    """Return whether configured raw-variance or raw/clipped-ratio gates fire."""
    if not config.get("enabled", False):
        return False
    raw = float(np.asarray(jax.device_get(metrics["variance_noclip"])).reshape(-1)[0])
    clipped = float(np.asarray(jax.device_get(metrics["variance"])).reshape(-1)[0])
    ratio = raw / max(abs(clipped), np.finfo(float).tiny)
    return (
        raw >= float(config.get("raw_variance_threshold", np.inf))
        or ratio >= float(config.get("raw_clipped_ratio_threshold", np.inf))
    )


def _tree_to_host(tree):
    return jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), tree)


def _distance_diagnostics(positions: Optional[np.ndarray]) -> dict:
    if positions is None:
        return {}
    positions = np.asarray(positions)
    flat = positions.reshape((-1,) + positions.shape[-2:])
    radii = np.linalg.norm(flat, axis=-1)
    result = {
        "electron_radius_min": float(radii.min()),
        "electron_radius_q001": float(np.quantile(radii, 0.001)),
        "electron_radius_q999": float(np.quantile(radii, 0.999)),
        "electron_radius_max": float(radii.max()),
    }
    if flat.shape[-2] > 1:
        displacement = flat[:, :, None, :] - flat[:, None, :, :]
        distance = np.linalg.norm(displacement, axis=-1)
        upper = np.triu_indices(flat.shape[-2], 1)
        pair = distance[:, upper[0], upper[1]]
        result.update(
            electron_pair_distance_min=float(pair.min()),
            electron_pair_distance_q001=float(np.quantile(pair, 0.001)),
            electron_pair_distance_q999=float(np.quantile(pair, 0.999)),
            electron_pair_distance_max=float(pair.max()),
        )
    return result


def maybe_save_burst(
    *,
    epoch: int,
    logdir: Optional[str],
    config: Mapping[str, Any],
    metrics: MutableMapping[str, Any],
    params,
    optimizer_state,
    data,
    key,
) -> Optional[str]:
    """Save a complete rare-event snapshot and return its directory, if triggered."""
    if not config.get("enabled", False):
        return None
    payload = pop_payload(metrics)
    if not should_capture(metrics, config):
        return None
    if logdir is None:
        raise ValueError("burst diagnostics require a logdir")

    root = config.get("output_dir", "burst_diagnostics")
    if not os.path.isabs(root):
        root = os.path.join(logdir, root)
    event_dir = os.path.join(root, f"epoch_{epoch + 1:08d}")
    os.makedirs(event_dir, exist_ok=False)

    host_payload = {name: np.asarray(jax.device_get(value)) for name, value in payload.items()}
    positions = host_payload.get("walker_coordinates")
    amplitudes = host_payload.get("walker_amplitudes")
    if positions is None:
        try:
            positions = np.asarray(jax.device_get(data["walker_data"]["position"]))
            amplitudes = np.asarray(jax.device_get(data["walker_data"]["amplitude"]))
        except (KeyError, TypeError):
            positions = None
    if positions is not None:
        host_payload["walker_coordinates"] = positions
    if amplitudes is not None:
        host_payload["walker_amplitudes"] = amplitudes
    np.savez(os.path.join(event_dir, "burst_payload.npz"), **host_payload)

    io.save_vmc_state(
        event_dir,
        "model_sampler_state.npz",
        (epoch, _tree_to_host(data), _tree_to_host(params),
         _tree_to_host(optimizer_state), np.asarray(jax.device_get(key))),
    )
    manifest = {
        "epoch": epoch + 1,
        "raw_variance": float(np.asarray(jax.device_get(metrics["variance_noclip"])).reshape(-1)[0]),
        "clipped_variance": float(np.asarray(jax.device_get(metrics["variance"])).reshape(-1)[0]),
        "payload_fields": sorted(host_payload),
        "hamiltonian_components_available": "hamiltonian_components" in host_payload,
        "distance_diagnostics": _distance_diagnostics(positions),
    }
    with open(os.path.join(event_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return event_dir
