"""JAX inference for FoveaVLA actions and future interaction boxes."""

from collections.abc import Sequence
import dataclasses
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from foveavla_client import base_policy as _base_policy
from foveavla_client import foveation
from typing_extensions import override

from foveavla import transforms as _transforms
from foveavla.models import model as _model
from foveavla.shared import array_typing as at
from foveavla.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


@dataclasses.dataclass(frozen=True)
class PendingFutureBbox:
    prefix: Any


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = foveation.with_legacy_fields(metadata or {})
        self._sample_actions_with_future_bbox = nnx_utils.module_jit(model.sample_actions_with_future_bbox)
        self._prepare_inference_prefix = nnx_utils.module_jit(model.prepare_inference_prefix)
        self._sample_actions_from_prepared_prefix = nnx_utils.module_jit(model.sample_actions_from_prepared_prefix)
        self._sample_future_bbox_from_prepared_prefix = nnx_utils.module_jit(
            model.sample_future_bbox_from_prepared_prefix
        )
        self._rng = jax.random.key(0) if rng is None else rng

    def _prepare_inference(self, obs: dict, noise: np.ndarray | None):
        # Transform a copy, then add the batch dimension expected by the model.
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            sample_kwargs["noise"] = noise[None, ...] if noise.ndim == 2 else noise
        return inputs, sample_rng, _model.Observation.from_dict(inputs), sample_kwargs

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs, rng, observation, kwargs = self._prepare_inference(obs, noise)
        started = time.monotonic()
        actions, bbox, valid = self._sample_actions_with_future_bbox(rng, observation, **kwargs)
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0]),
            {
                "state": inputs["state"],
                "actions": actions,
                "future_interaction_box": bbox,
                "future_interaction_box_valid": valid,
            },
        )
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000}
        return foveation.with_legacy_fields(outputs)

    def infer_actions_with_pending_bbox(self, obs: dict, *, noise: np.ndarray | None = None):
        """Return actions first; decode the next interaction box while the robot executes them."""
        inputs, rng, observation, kwargs = self._prepare_inference(obs, noise)
        started = time.monotonic()
        prefix = self._prepare_inference_prefix(observation)
        actions = self._sample_actions_from_prepared_prefix(rng, prefix, **kwargs)
        outputs = jax.tree.map(lambda x: np.asarray(x[0]), {"state": inputs["state"], "actions": actions})
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"actions_infer_ms": (time.monotonic() - started) * 1000}
        return outputs, PendingFutureBbox(prefix)

    def infer_future_bbox(self, pending: PendingFutureBbox) -> dict:
        started = time.monotonic()
        bbox, valid = self._sample_future_bbox_from_prepared_prefix(pending.prefix)
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0]),
            {"future_interaction_box": bbox, "future_interaction_box_valid": valid},
        )
        outputs["policy_timing"] = {"bbox_infer_ms": (time.monotonic() - started) * 1000}
        return foveation.with_legacy_fields(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
