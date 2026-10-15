"""FoveaVLA: shared RGB/depth SigLIP tokens, π₀.₅ actions, and t+8 foveal decoding."""

import logging
from typing import Any, NamedTuple

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from foveavla.models import future_bbox
from foveavla.models import model as _model
from foveavla.models import foveavla_config
import foveavla.models.gemma as _gemma
import foveavla.models.siglip as _siglip
from foveavla.shared import array_typing as at

logger = logging.getLogger("foveavla")

METRIC_DEPTH_NEAR_M = 0.2
METRIC_DEPTH_FAR_M = 3.0
METRIC_DEPTH_QUALITY_MIN = 0.1
METRIC_DEPTH_PATCH_MIN_VALID_FRACTION = 0.25
_TURBO_POLYNOMIAL_COEFFICIENTS = (
    (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
    (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
    (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
)


class PreparedInferencePrefix(NamedTuple):
    observation: _model.Observation
    tokens: jax.Array
    mask: jax.Array
    ar_mask: jax.Array
    output: jax.Array
    kv_cache: Any


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_joint_attn_mask(prefix_mask, prefix_ar_mask, action_prefix_mask, suffix_mask, suffix_ar_mask):
    """Build independent VLM-bbox and action attention branches in one forward pass."""
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    batch_size, prefix_len = prefix_mask.shape
    suffix_len = suffix_mask.shape[1]

    combined_mask = jnp.zeros((batch_size, prefix_len + suffix_len, prefix_len + suffix_len), dtype=bool)
    combined_mask = combined_mask.at[:, :prefix_len, :prefix_len].set(prefix_attn_mask)

    action_input_mask = jnp.concatenate([action_prefix_mask, suffix_mask], axis=1)
    action_ar_mask = jnp.concatenate([jnp.zeros_like(action_prefix_mask), suffix_ar_mask], axis=1)
    action_attn_mask = make_attn_mask(action_input_mask, action_ar_mask)
    return combined_mask.at[:, prefix_len:, :].set(action_attn_mask[:, prefix_len:, :])


def make_joint_positions(prefix_mask, action_prefix_mask, suffix_mask):
    """Use all VLM tokens for bbox positions but exclude bbox tokens from action positions."""
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    suffix_positions = jnp.sum(action_prefix_mask, axis=1, keepdims=True) + jnp.cumsum(suffix_mask, axis=1) - 1
    return jnp.concatenate([prefix_positions, suffix_positions], axis=1).astype(jnp.int32)


def compute_future_bbox_loss(bbox_logits, target_classes, target_mask):
    """Compute masked four-location-token loss and training-time geometry metrics."""
    bbox_logits = bbox_logits.astype(jnp.float32)
    target_mask = jnp.asarray(target_mask, dtype=jnp.bool_)
    token_loss = -jnp.take_along_axis(jax.nn.log_softmax(bbox_logits, axis=-1), target_classes[..., None], axis=-1)[
        ..., 0
    ]

    target_box_mask = target_mask.reshape(target_mask.shape[0], -1, 4)
    box_valid = jnp.all(target_box_mask, axis=-1)
    valid_token_mask = jnp.broadcast_to(box_valid[..., None], target_box_mask.shape).reshape(target_mask.shape)
    sample_valid = jnp.any(box_valid, axis=1)
    valid_token_count = jnp.sum(valid_token_mask, axis=1)
    raw_sample_loss = jnp.sum(token_loss * valid_token_mask, axis=1) / jnp.clip(valid_token_count, 1)
    valid_count = jnp.sum(sample_valid)
    sample_loss = raw_sample_loss * bbox_logits.shape[0] / jnp.clip(valid_count, 1)

    predictions = jnp.argmax(bbox_logits, axis=-1)
    prediction_classes = predictions.reshape(predictions.shape[0], -1, 4)
    target_box_classes = target_classes.reshape(target_classes.shape[0], -1, 4)
    prediction_boxes = _location_classes_to_source_xyxy(prediction_classes)
    target_boxes = _location_classes_to_source_xyxy(target_box_classes)
    bbox_iou = _box_iou_xyxy(prediction_boxes, target_boxes)
    invalid = jnp.logical_or(
        prediction_classes[..., 0] >= prediction_classes[..., 2],
        prediction_classes[..., 1] >= prediction_classes[..., 3],
    )
    valid_box_count = jnp.sum(box_valid)

    metrics = {
        "bbox_loss": jnp.mean(sample_loss),
        "bbox_token_accuracy": jnp.sum((predictions == target_classes) * valid_token_mask)
        / jnp.clip(jnp.sum(valid_token_mask), 1),
        "bbox_iou": jnp.sum(bbox_iou * box_valid) / jnp.clip(valid_box_count, 1),
        "bbox_invalid_rate": jnp.sum(invalid * box_valid) / jnp.clip(valid_box_count, 1),
        "bbox_valid_fraction": jnp.mean(box_valid),
    }
    return sample_loss, metrics


def _location_classes_to_source_xyxy(classes):
    coords = classes.astype(jnp.float32) / (future_bbox.NUM_LOCATION_BINS - 1)
    y1, x1, y2, x2 = jnp.moveaxis(coords, -1, 0)
    ratio = max(
        future_bbox.RAW_IMAGE_WIDTH / future_bbox.MODEL_IMAGE_SIZE,
        future_bbox.RAW_IMAGE_HEIGHT / future_bbox.MODEL_IMAGE_SIZE,
    )
    resized_height = int(future_bbox.RAW_IMAGE_HEIGHT / ratio)
    resized_width = int(future_bbox.RAW_IMAGE_WIDTH / ratio)
    pad_y = (future_bbox.MODEL_IMAGE_SIZE - resized_height) // 2
    pad_x = (future_bbox.MODEL_IMAGE_SIZE - resized_width) // 2
    source_x1 = (x1 * future_bbox.MODEL_IMAGE_SIZE - pad_x) * ratio / future_bbox.RAW_IMAGE_WIDTH
    source_x2 = (x2 * future_bbox.MODEL_IMAGE_SIZE - pad_x) * ratio / future_bbox.RAW_IMAGE_WIDTH
    source_y1 = (y1 * future_bbox.MODEL_IMAGE_SIZE - pad_y) * ratio / future_bbox.RAW_IMAGE_HEIGHT
    source_y2 = (y2 * future_bbox.MODEL_IMAGE_SIZE - pad_y) * ratio / future_bbox.RAW_IMAGE_HEIGHT
    return jnp.clip(jnp.stack([source_x1, source_y1, source_x2, source_y2], axis=-1), 0.0, 1.0)


def _box_iou_xyxy(boxes_a, boxes_b):
    intersection_min = jnp.maximum(boxes_a[..., :2], boxes_b[..., :2])
    intersection_max = jnp.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection_area = jnp.prod(jnp.maximum(intersection_max - intersection_min, 0.0), axis=-1)
    area_a = jnp.prod(jnp.maximum(boxes_a[..., 2:] - boxes_a[..., :2], 0.0), axis=-1)
    area_b = jnp.prod(jnp.maximum(boxes_b[..., 2:] - boxes_b[..., :2], 0.0), axis=-1)
    union = area_a + area_b - intersection_area
    return intersection_area / jnp.maximum(union, 1e-8)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _turbo_colormap(value):
    """Apply Google's continuous polynomial approximation of the Turbo colormap."""
    value = jnp.clip(jnp.asarray(value, dtype=jnp.float32), 0.0, 1.0)
    coefficients = jnp.asarray(_TURBO_POLYNOMIAL_COEFFICIENTS, dtype=jnp.float32)
    result = jnp.zeros((*value.shape, 3), dtype=jnp.float32)
    for coefficient in coefficients[:, ::-1].T:
        result = result * value[..., None] + coefficient
    return jnp.clip(result, 0.0, 1.0)


def _metric_depth_to_siglip_image(depth_m, quality, *, patch_size: int = 14):
    """Colorize fixed-range metric Z-depth and mask unreliable SigLIP patches."""
    depth_m = jnp.asarray(depth_m, dtype=jnp.float32)
    quality = jnp.asarray(quality, dtype=jnp.float32)
    if depth_m.ndim != 3 or depth_m.shape[1:] != (224, 224):
        raise ValueError(f"Expected metric depth shape [batch, 224, 224], got {depth_m.shape}")
    if quality.ndim != 3 or quality.shape != depth_m.shape:
        raise ValueError(f"Expected depth quality shape {depth_m.shape}, got {quality.shape}")

    valid = (
        jnp.isfinite(depth_m)
        & jnp.isfinite(quality)
        & (depth_m >= METRIC_DEPTH_NEAR_M)
        & (depth_m <= METRIC_DEPTH_FAR_M)
        & (quality >= METRIC_DEPTH_QUALITY_MIN)
    )
    safe_depth = jnp.clip(depth_m, METRIC_DEPTH_NEAR_M, METRIC_DEPTH_FAR_M)
    normalized_log_depth = jnp.log(safe_depth / METRIC_DEPTH_NEAR_M) / jnp.log(METRIC_DEPTH_FAR_M / METRIC_DEPTH_NEAR_M)
    image = _turbo_colormap(1.0 - normalized_log_depth) * 2.0 - 1.0
    image = jnp.where(valid[..., None], image, -1.0)

    grid_size = depth_m.shape[1] // patch_size
    patch_valid_fraction = einops.reduce(
        valid.astype(jnp.float32),
        "b (h ph) (w pw) -> b (h w)",
        "mean",
        h=grid_size,
        w=grid_size,
        ph=patch_size,
        pw=patch_size,
    )
    patch_valid = patch_valid_fraction >= METRIC_DEPTH_PATCH_MIN_VALID_FRACTION
    return image, patch_valid


class FoveaVLA(_model.BaseModel):
    def __init__(self, config: foveavla_config.FoveaVLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.image_keys = config.image_keys
        self.depth_image_keys = config.depth_image_keys
        self.depth_view_dropout_prob = config.depth_view_dropout_prob
        self.future_bbox_count = config.future_bbox_count
        self.bbox_loss_weight = config.bbox_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

    def _validate_depth_inputs(self, obs: _model.Observation):
        expected = set(self.depth_image_keys)
        depth_keys = set() if obs.depthmaps is None else set(obs.depthmaps)
        quality_keys = set() if obs.depthmap_quality is None else set(obs.depthmap_quality)
        if depth_keys != expected or quality_keys != expected:
            raise ValueError(
                f"Expected metric depth and quality for {sorted(expected)}, "
                f"got {sorted(depth_keys)} and {sorted(quality_keys)}"
            )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, rng=None, train: bool = False
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        """Encode RGB, then aligned metric depth, then the task/state/foveal prompt."""
        self._validate_depth_inputs(obs)
        tokens, input_mask, ar_mask = [], [], []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))

        assert obs.depthmaps is not None and obs.depthmap_quality is not None
        if train and self.depth_view_dropout_prob:
            if rng is None:
                raise ValueError("Depth view dropout requires an RNG")
            view_keep = jax.random.bernoulli(
                rng,
                p=1.0 - self.depth_view_dropout_prob,
                shape=(obs.state.shape[0], len(self.depth_image_keys)),
            )
        else:
            view_keep = jnp.ones((obs.state.shape[0], len(self.depth_image_keys)), dtype=jnp.bool_)
        for view_index, image_key in enumerate(self.depth_image_keys):
            depth_image, patch_valid = _metric_depth_to_siglip_image(
                obs.depthmaps[image_key], obs.depthmap_quality[image_key]
            )
            patch_valid &= obs.image_masks[image_key][:, None]
            patch_valid &= view_keep[:, view_index, None]
            depth_tokens, _ = self.PaliGemma.img(depth_image, train=False, token_mask=patch_valid)
            tokens.append(depth_tokens)
            input_mask.append(patch_valid)
            ar_mask.append(jnp.zeros(depth_tokens.shape[:2], dtype=jnp.bool_))

        assert obs.tokenized_prompt is not None and obs.tokenized_prompt_mask is not None
        assert obs.token_ar_mask is not None, "Future bbox token autoregressive mask is required"
        tokens.append(self.PaliGemma.llm(obs.tokenized_prompt, method="embed"))
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(jnp.asarray(obs.token_ar_mask, dtype=jnp.bool_))
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(input_mask, axis=1), jnp.concatenate(ar_mask, axis=1)

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # π₀.₅ conditions the action expert's adaptive RMSNorm on the flow timestep.
        time_emb = nnx.swish(self.time_mlp_in(time_emb))
        adarms_cond = nnx.swish(self.time_mlp_out(time_emb))
        action_expert_tokens = action_tokens
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        action_ar_mask = jnp.zeros(action_expert_tokens.shape[:2], dtype=jnp.bool_)
        action_ar_mask = action_ar_mask.at[:, 0].set(True)
        ar_mask.append(action_ar_mask)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask, adarms_cond

    def _action_prefix_mask(self, prefix_mask, prefix_ar_mask):
        return jnp.logical_and(prefix_mask, jnp.logical_not(prefix_ar_mask))

    def _compute_bbox_loss(self, observation, prefix_out):
        assert observation.tokenized_prompt is not None
        assert observation.tokenized_prompt_mask is not None
        assert observation.token_loss_mask is not None, "Future bbox token loss mask is required"

        batch_size = observation.tokenized_prompt.shape[0]
        text_len = observation.tokenized_prompt.shape[1]
        coordinate_count = 4 * self.future_bbox_count
        text_target_indices = (
            jnp.sum(observation.tokenized_prompt_mask, axis=1, keepdims=True)
            - coordinate_count
            + jnp.arange(coordinate_count, dtype=jnp.int32)[None, :]
        )
        batch_indices = jnp.arange(batch_size)[:, None]
        target_token_ids = observation.tokenized_prompt[batch_indices, text_target_indices]
        target_mask = observation.token_loss_mask[batch_indices, text_target_indices]

        image_token_len = prefix_out.shape[1] - text_len
        hidden_indices = image_token_len + text_target_indices - 1
        bbox_hidden = prefix_out[batch_indices, hidden_indices]
        location_token_ids = jnp.arange(
            future_bbox.LOCATION_TOKEN_START_ID,
            future_bbox.LOCATION_TOKEN_START_ID + future_bbox.NUM_LOCATION_BINS,
            dtype=jnp.int32,
        )
        bbox_logits = self.PaliGemma.llm(bbox_hidden, location_token_ids, method="decode_subset").astype(jnp.float32)
        target_classes = target_token_ids - future_bbox.LOCATION_TOKEN_START_ID
        return compute_future_bbox_loss(bbox_logits, target_classes, target_mask)

    def _decode_future_bbox(self, prefix_out, prefix_mask, kv_cache):
        """Greedily decode one or more four-location-token boxes from an inference prefix."""
        batch_size = prefix_mask.shape[0]
        batch_indices = jnp.arange(batch_size)
        sequence_indices = jnp.arange(prefix_mask.shape[1], dtype=jnp.int32)
        last_prefix_indices = jnp.max(jnp.where(prefix_mask, sequence_indices[None, :], -1), axis=1)
        bbox_hidden = prefix_out[batch_indices, last_prefix_indices]
        location_token_ids = jnp.arange(
            future_bbox.LOCATION_TOKEN_START_ID,
            future_bbox.LOCATION_TOKEN_START_ID + future_bbox.NUM_LOCATION_BINS,
            dtype=jnp.int32,
        )

        bbox_count = getattr(self, "future_bbox_count", 1)
        coordinate_count = 4 * bbox_count
        location_classes = []
        for coordinate_index in range(coordinate_count):
            bbox_logits = self.PaliGemma.llm(bbox_hidden, location_token_ids, method="decode_subset").astype(
                jnp.float32
            )
            predicted_class = jnp.argmax(bbox_logits, axis=-1).astype(jnp.int32)
            location_classes.append(predicted_class)
            if coordinate_index == coordinate_count - 1:
                break

            predicted_token = predicted_class[:, None] + future_bbox.LOCATION_TOKEN_START_ID
            token_embedding = self.PaliGemma.llm(predicted_token, method="embed")
            generated_mask = jnp.ones((batch_size, coordinate_index + 1), dtype=jnp.bool_)
            decode_mask = jnp.concatenate([prefix_mask, generated_mask], axis=1)[:, None, :]
            positions = (jnp.sum(prefix_mask, axis=1) + coordinate_index)[:, None].astype(jnp.int32)
            (bbox_out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None],
                mask=decode_mask,
                positions=positions,
                kv_cache=kv_cache,
            )
            bbox_hidden = bbox_out[:, 0]

        classes = jnp.stack(location_classes, axis=-1).reshape(batch_size, bbox_count, 4)
        bbox = _location_classes_to_source_xyxy(classes)
        valid = jnp.logical_and(bbox[..., 0] < bbox[..., 2], bbox[..., 1] < bbox[..., 3])
        if bbox_count == 1:
            return bbox[:, 0], valid[:, 0]
        return bbox, valid

    def _compute_loss(self, rng, observation, actions, *, train):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        depth_rng = jax.random.fold_in(rng, 0x3D)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, rng=depth_rng, train=train)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        action_prefix_mask = self._action_prefix_mask(prefix_mask, prefix_ar_mask)
        attn_mask = make_joint_attn_mask(prefix_mask, prefix_ar_mask, action_prefix_mask, suffix_mask, suffix_ar_mask)
        positions = make_joint_positions(prefix_mask, action_prefix_mask, suffix_mask)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        metrics = {"action_loss": jnp.mean(action_loss)}

        bbox_loss, bbox_metrics = self._compute_bbox_loss(observation, prefix_out)
        metrics.update(bbox_metrics)
        return action_loss + self.bbox_loss_weight * bbox_loss[:, None], metrics

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._compute_loss(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        return self._compute_loss(rng, observation, actions, train=train)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        prefix = self.prepare_inference_prefix(observation)
        return self.sample_actions_from_prepared_prefix(
            rng,
            prefix,
            num_steps=num_steps,
            noise=noise,
        )

    def prepare_inference_prefix(self, observation: _model.Observation) -> PreparedInferencePrefix:
        """Encode an observation once for staged action and future-bbox inference."""
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=self.image_keys)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return PreparedInferencePrefix(
            observation=observation,
            tokens=prefix_tokens,
            mask=prefix_mask,
            ar_mask=prefix_ar_mask,
            output=prefix_out,
            kv_cache=kv_cache,
        )

    def sample_actions_from_prepared_prefix(
        self,
        rng: at.KeyArrayLike,
        prefix: PreparedInferencePrefix,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        return self._sample_actions_from_prefix(
            rng,
            prefix.observation,
            prefix.tokens,
            prefix.mask,
            prefix.ar_mask,
            prefix.kv_cache,
            num_steps=num_steps,
            noise=noise,
        )

    def sample_future_bbox_from_prepared_prefix(self, prefix: PreparedInferencePrefix) -> tuple[jax.Array, jax.Array]:
        return self._decode_future_bbox(prefix.output, prefix.mask, prefix.kv_cache)

    def sample_actions_with_future_bbox(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ):
        """Sample actions and decode the t+8 interaction box from one shared prefix."""

        prefix = self.prepare_inference_prefix(observation)
        actions = self.sample_actions_from_prepared_prefix(
            rng,
            prefix,
            num_steps=num_steps,
            noise=noise,
        )
        bbox, bbox_valid = self.sample_future_bbox_from_prepared_prefix(prefix)
        return actions, bbox, bbox_valid

    def sample_future_bbox(self, observation: _model.Observation):
        """Greedily decode the future interaction box without sampling actions."""

        prefix = self.prepare_inference_prefix(observation)
        return self.sample_future_bbox_from_prepared_prefix(prefix)

    def _sample_actions_from_prefix(
        self,
        rng,
        observation,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        kv_cache,
        *,
        num_steps,
        noise,
    ):
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        action_prefix_mask = self._action_prefix_mask(prefix_mask, prefix_ar_mask)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(action_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(action_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
