"""Function-preserving prolongation between nested FermiNet architectures.

The routines in this module embed a trained, smaller FermiNet into a wider or
deeper FermiNet without changing the represented wavefunction.  They are kept
separate from model construction so that ordinary FermiNet runs are unchanged.

The implementation is semantic rather than generic zero padding.  In
particular, the inputs to the mixed one-electron kernels are concatenated by
spin, so their old coordinates are not a contiguous prefix after widening.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping, Optional

import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze


def _plain(value: Any) -> Any:
    """Convert ConfigDict-like values to ordinary containers for comparison."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


def _dense_specs(model_config: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(width) for width in block)
        for block in model_config.backflow.ndense_list
    )


def validate_nested_ferminet_configs(
    coarse_model_config: Any,
    fine_model_config: Any,
) -> None:
    """Validate the currently supported strict FermiNet nesting relation.

    The first implementation deliberately keeps the determinant count fixed.
    Widening and residual-depth growth can be both function preserving and
    differentiably safe.  Adding exactly-zero determinants through ``slogdet``
    would place the model at a singular matrix, so determinant growth requires a
    separate, explicit determinant-gating implementation.
    """
    if coarse_model_config.type != "ferminet" or fine_model_config.type != "ferminet":
        raise ValueError("multilevel prolongation currently supports FermiNet only")

    invariant_fields = (
        "input_streams",
        "full_det",
        "isotropic_decay",
        "envelope_softening",
        "orbitals_use_bias",
        "include_cusp_jastrow",
    )
    for field in invariant_fields:
        if _plain(coarse_model_config[field]) != _plain(fine_model_config[field]):
            raise ValueError(f"coarse and fine FermiNets must have the same {field}")

    coarse_backflow = coarse_model_config.backflow
    fine_backflow = fine_model_config.backflow
    backflow_invariants = (
        "activation_fn",
        "use_bias",
        "one_electron_skip",
        "one_electron_skip_scale",
        "two_electron_skip",
        "two_electron_skip_scale",
    )
    for field in backflow_invariants:
        if _plain(coarse_backflow[field]) != _plain(fine_backflow[field]):
            raise ValueError(f"coarse and fine backflows must have the same {field}")

    if int(coarse_model_config.ndeterminants) != int(fine_model_config.ndeterminants):
        raise ValueError(
            "strict, trainable determinant-count growth is not yet supported; "
            "keep ndeterminants fixed while widening/deepening"
        )

    coarse_specs = _dense_specs(coarse_model_config)
    fine_specs = _dense_specs(fine_model_config)
    if len(coarse_specs) > len(fine_specs):
        raise ValueError("the fine FermiNet cannot have fewer residual blocks")
    for block, coarse_spec in enumerate(coarse_specs):
        fine_spec = fine_specs[block]
        if coarse_spec[0] > fine_spec[0]:
            raise ValueError(f"one-electron width shrinks in residual block {block}")
        if len(coarse_spec) > 1:
            if len(fine_spec) < 2:
                raise ValueError(
                    f"two-electron layer is removed in residual block {block}"
                )
            if coarse_spec[1] > fine_spec[1]:
                raise ValueError(
                    f"two-electron width shrinks in residual block {block}"
                )


def _copy_kernel(
    source_dense: Optional[Mapping[str, Any]],
    target_dense: MutableMapping[str, Any],
    old_output_width: int,
    *,
    nspins: int,
    spin_concatenated_input: bool,
    keep_trainable_sidecar: bool,
    sidecar_scale: float,
) -> None:
    """Embed one Dense map while preventing new inputs from changing old outputs."""
    target_kernel = target_dense["kernel"]
    result = (
        jnp.asarray(sidecar_scale, dtype=target_kernel.dtype) * target_kernel
        if keep_trainable_sidecar
        else jnp.zeros_like(target_kernel)
    )
    # Old output coordinates must be independent of every newly added input.
    result = result.at[:, :old_output_width].set(0)

    if source_dense is not None:
        source_kernel = source_dense["kernel"]
        if source_kernel.shape[1] != old_output_width:
            raise ValueError("source Dense output width disagrees with the level spec")
        if spin_concatenated_input:
            if source_kernel.shape[0] % nspins or target_kernel.shape[0] % nspins:
                raise ValueError(
                    "spin-concatenated Dense input is not divisible by nspins"
                )
            source_width = source_kernel.shape[0] // nspins
            target_width = target_kernel.shape[0] // nspins
            if source_width > target_width:
                raise ValueError("spin-concatenated Dense input shrinks")
            for spin in range(nspins):
                source_slice = slice(spin * source_width, (spin + 1) * source_width)
                target_slice = slice(
                    spin * target_width, spin * target_width + source_width
                )
                result = result.at[target_slice, :old_output_width].set(
                    source_kernel[source_slice, :]
                )
        else:
            if source_kernel.shape[0] > target_kernel.shape[0]:
                raise ValueError("Dense input width shrinks")
            result = result.at[: source_kernel.shape[0], :old_output_width].set(
                source_kernel
            )
    target_dense["kernel"] = result

    if "bias" in target_dense:
        target_bias = target_dense["bias"]
        bias_result = (
            jnp.asarray(sidecar_scale, dtype=target_bias.dtype) * target_bias
            if keep_trainable_sidecar
            else jnp.zeros_like(target_bias)
        )
        bias_result = bias_result.at[:old_output_width].set(0)
        if source_dense is not None and "bias" in source_dense:
            bias_result = bias_result.at[:old_output_width].set(source_dense["bias"])
        target_dense["bias"] = bias_result


def _copy_one_electron_layer(
    source_layer: Optional[Mapping[str, Any]],
    target_layer: MutableMapping[str, Any],
    old_output_width: int,
    *,
    nspins: int,
    keep_trainable_sidecar: bool,
    sidecar_scale: float,
) -> None:
    for name, spin_concatenated in (
        ("_unmixed_dense", False),
        ("_mixed_dense", True),
        ("_dense_2e", True),
    ):
        _copy_kernel(
            None if source_layer is None else source_layer[name],
            target_layer[name],
            old_output_width,
            nspins=nspins,
            spin_concatenated_input=spin_concatenated,
            keep_trainable_sidecar=keep_trainable_sidecar,
            sidecar_scale=sidecar_scale,
        )


def _copy_two_electron_layer(
    source_layer: Optional[Mapping[str, Any]],
    target_layer: MutableMapping[str, Any],
    old_output_width: int,
    *,
    nspins: int,
    keep_trainable_sidecar: bool,
    sidecar_scale: float,
) -> None:
    _copy_kernel(
        None if source_layer is None else source_layer["_dense"],
        target_layer["_dense"],
        old_output_width,
        nspins=nspins,
        spin_concatenated_input=False,
        keep_trainable_sidecar=keep_trainable_sidecar,
        sidecar_scale=sidecar_scale,
    )


def _copy_orbital_layer(
    source: Mapping[str, Any],
    target: MutableMapping[str, Any],
) -> None:
    """Copy the old orbital head and disconnect all newly added hidden features."""
    for name, source_child in source.items():
        if name == "SplitDense_0":
            continue
        if name not in target:
            raise ValueError(f"fine orbital layer is missing {name}")
        for parameter_name, source_value in source_child.items():
            target_value = target[name][parameter_name]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    "orbital envelope shape changed; determinant-count growth is "
                    "not supported by this prolongation"
                )
            target[name][parameter_name] = jnp.array(source_value)

    source_split = source["SplitDense_0"]
    target_split = target["SplitDense_0"]
    for dense_name, source_dense in source_split.items():
        target_dense = target_split[dense_name]
        source_kernel = source_dense["kernel"]
        target_kernel = target_dense["kernel"]
        if source_kernel.shape[1] != target_kernel.shape[1]:
            raise ValueError("orbital output width changed")
        if source_kernel.shape[0] > target_kernel.shape[0]:
            raise ValueError("orbital input width shrinks")
        embedded_kernel = jnp.zeros_like(target_kernel)
        embedded_kernel = embedded_kernel.at[: source_kernel.shape[0], :].set(
            source_kernel
        )
        target_dense["kernel"] = embedded_kernel
        if "bias" in source_dense:
            if source_dense["bias"].shape != target_dense["bias"].shape:
                raise ValueError("orbital bias shape changed")
            target_dense["bias"] = jnp.array(source_dense["bias"])


def prolongate_ferminet_params(
    coarse_params: Any,
    fine_template_params: Any,
    coarse_model_config: Any,
    fine_model_config: Any,
    *,
    nspins: int,
    keep_trainable_sidecar: bool = True,
    sidecar_scale: float = 1.0,
) -> Any:
    """Embed coarse FermiNet parameters into a fine FermiNet PyTree.

    When ``keep_trainable_sidecar`` is true, randomly initialized fine-only
    hidden features are retained, but every path from them into an old output is
    set to zero.  The wavefunction is unchanged, while the zero orbital coupling
    receives a nonzero gradient and lets the added features enter training.  The
    false mode is the linear tangent embedding used for optimizer memories.
    """
    if nspins <= 0:
        raise ValueError("nspins must be positive")
    if sidecar_scale < 0:
        raise ValueError("sidecar_scale must be nonnegative")
    validate_nested_ferminet_configs(coarse_model_config, fine_model_config)

    target_was_frozen = isinstance(fine_template_params, FrozenDict)
    source = (
        unfreeze(coarse_params)
        if isinstance(coarse_params, FrozenDict)
        else copy.deepcopy(coarse_params)
    )
    target = (
        unfreeze(fine_template_params)
        if target_was_frozen
        else copy.deepcopy(fine_template_params)
    )
    if not keep_trainable_sidecar:
        target = _tree_zeros(target)

    source_root = source["params"]
    target_root = target["params"]
    source_backflow = source_root["backflow"]
    target_backflow = target_root["backflow"]
    coarse_specs = _dense_specs(coarse_model_config)
    fine_specs = _dense_specs(fine_model_config)

    old_one_width: Optional[int] = None
    old_two_width: Optional[int] = None
    for block_index, fine_spec in enumerate(fine_specs):
        block_name = f"residual_blocks_{block_index}"
        target_block = target_backflow[block_name]
        source_block = source_backflow.get(block_name)
        source_spec = (
            coarse_specs[block_index]
            if block_index < len(coarse_specs)
            else None
        )

        target_one_dense = target_block["one_electron_layer"]["_unmixed_dense"]
        target_one_input = target_one_dense["kernel"].shape[0]
        target_one_output = target_one_dense["kernel"].shape[1]
        if source_spec is not None:
            source_one_dense = source_block["one_electron_layer"]["_unmixed_dense"]
            source_one_input = source_one_dense["kernel"].shape[0]
            old_one_output = source_one_dense["kernel"].shape[1]
            coarse_skip = (
                bool(coarse_model_config.backflow.one_electron_skip)
                and source_one_input == old_one_output
            )
            fine_skip = (
                bool(fine_model_config.backflow.one_electron_skip)
                and target_one_input == target_one_output
            )
            if coarse_skip != fine_skip:
                raise ValueError(
                    f"one-electron skip behavior changes in block {block_index}"
                )
        else:
            if old_one_width is None:
                raise ValueError("a coarse FermiNet must contain at least one block")
            old_one_output = old_one_width
            if (
                not bool(fine_model_config.backflow.one_electron_skip)
                or target_one_input != target_one_output
            ):
                raise ValueError(
                    f"new residual block {block_index} is not an exact identity at zero"
                )
        if old_one_output > target_one_output:
            raise ValueError(
                "fine one-electron output cannot contain the coarse output"
            )
        _copy_one_electron_layer(
            None if source_block is None else source_block["one_electron_layer"],
            target_block["one_electron_layer"],
            old_one_output,
            nspins=nspins,
            keep_trainable_sidecar=keep_trainable_sidecar,
            sidecar_scale=sidecar_scale,
        )
        old_one_width = old_one_output

        # The one-electron layer always exposes the current two-electron input
        # width through its spin-concatenated _dense_2e kernel.
        if source_block is not None:
            old_two_input = (
                source_block["one_electron_layer"]["_dense_2e"]["kernel"].shape[0]
                // nspins
            )
            if old_two_width is None:
                old_two_width = old_two_input

        source_two = (
            None
            if source_block is None
            else source_block.get("two_electron_layer")
        )
        target_two = target_block.get("two_electron_layer")
        if source_two is not None and target_two is None:
            raise ValueError(f"two-electron layer is removed in block {block_index}")
        if target_two is not None:
            target_two_dense = target_two["_dense"]
            target_two_input = target_two_dense["kernel"].shape[0]
            target_two_output = target_two_dense["kernel"].shape[1]
            if source_two is not None:
                source_two_dense = source_two["_dense"]
                source_two_input = source_two_dense["kernel"].shape[0]
                old_two_output = source_two_dense["kernel"].shape[1]
                coarse_skip = (
                    bool(coarse_model_config.backflow.two_electron_skip)
                    and source_two_input == old_two_output
                )
                fine_skip = (
                    bool(fine_model_config.backflow.two_electron_skip)
                    and target_two_input == target_two_output
                )
                if coarse_skip != fine_skip:
                    raise ValueError(
                        f"two-electron skip behavior changes in block {block_index}"
                    )
            else:
                if old_two_width is None:
                    raise ValueError("cannot infer the coarse two-electron width")
                old_two_output = old_two_width
                if (
                    not bool(fine_model_config.backflow.two_electron_skip)
                    or target_two_input != target_two_output
                ):
                    raise ValueError(
                        f"new two-electron block {block_index} is not an exact identity"
                    )
            if old_two_output > target_two_output:
                raise ValueError(
                    "fine two-electron output cannot contain the coarse output"
                )
            _copy_two_electron_layer(
                source_two,
                target_two,
                old_two_output,
                nspins=nspins,
                keep_trainable_sidecar=keep_trainable_sidecar,
                sidecar_scale=sidecar_scale,
            )
            old_two_width = old_two_output

    _copy_orbital_layer(
        source_root["FermiNetOrbitalLayer_0"],
        target_root["FermiNetOrbitalLayer_0"],
    )

    handled = {"backflow", "FermiNetOrbitalLayer_0"}
    for name, source_subtree in source_root.items():
        if name in handled:
            continue
        if name not in target_root:
            raise ValueError(f"fine FermiNet is missing parameter subtree {name}")
        if _shape_tree(source_subtree) != _shape_tree(target_root[name]):
            raise ValueError(f"unsupported shape change in parameter subtree {name}")
        target_root[name] = copy.deepcopy(source_subtree)

    return freeze(target) if target_was_frozen else target


def prolongate_ferminet_tangent(
    coarse_tangent: Any,
    fine_params: Any,
    coarse_model_config: Any,
    fine_model_config: Any,
    *,
    nspins: int,
) -> Any:
    """Apply the linear zero-extension map to a parameter-space tangent."""
    return prolongate_ferminet_params(
        coarse_tangent,
        fine_params,
        coarse_model_config,
        fine_model_config,
        nspins=nspins,
        keep_trainable_sidecar=False,
        sidecar_scale=0.0,
    )


def ferminet_flat_prolongation_indices(
    coarse_params: Any,
    fine_params: Any,
    coarse_model_config: Any,
    fine_model_config: Any,
    *,
    nspins: int,
) -> jnp.ndarray:
    """Return fine flat-vector rows corresponding to each coarse parameter.

    If ``indices = ferminet_flat_prolongation_indices(...)``, a matrix whose
    first axis is the coarse flattened parameter axis can be embedded with
    ``fine_matrix.at[indices].set(coarse_matrix)``.  This is used for WSSR's
    low-rank history factors without materializing one tangent PyTree per rank.
    """
    next_index = 1
    index_leaves = []
    coarse_treedef = jax.tree_util.tree_structure(coarse_params)
    for leaf in jax.tree_util.tree_leaves(coarse_params):
        size = leaf.size
        index_leaves.append(
            jnp.arange(next_index, next_index + size, dtype=jnp.int32).reshape(
                leaf.shape
            )
        )
        next_index += size
    coarse_indices = jax.tree_util.tree_unflatten(coarse_treedef, index_leaves)
    fine_index_template = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(leaf.shape, dtype=jnp.int32), fine_params
    )
    embedded_indices = prolongate_ferminet_tangent(
        coarse_indices,
        fine_index_template,
        coarse_model_config,
        fine_model_config,
        nspins=nspins,
    )
    embedded_flat = jnp.concatenate(
        [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(embedded_indices)]
    )
    fine_rows = jnp.flatnonzero(embedded_flat, size=next_index - 1)
    labels = embedded_flat[fine_rows]
    order = jnp.argsort(labels)
    expected = jnp.arange(1, next_index, dtype=labels.dtype)
    if not bool(jnp.array_equal(labels[order], expected)):
        raise ValueError("parameter prolongation is not a one-to-one coordinate map")
    return fine_rows[order]


def _tree_zeros(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return type(tree)((key, _tree_zeros(value)) for key, value in tree.items())
    return jnp.zeros_like(tree)


def _shape_tree(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {key: _shape_tree(value) for key, value in tree.items()}
    return tuple(tree.shape)
