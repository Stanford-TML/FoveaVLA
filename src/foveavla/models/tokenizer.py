import logging

import numpy as np
import sentencepiece

from foveavla.models import future_bbox
import foveavla.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize_future_bbox(
        self,
        prompt: str,
        state: np.ndarray,
        bbox: np.ndarray | None = None,
        *,
        bbox_mask: bool | np.ndarray = True,
        bbox_count: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a PI0.5 prompt followed by the fixed future-bbox decoder branch."""
        prompt_tokens = self._tokenize_prompt(prompt, state)
        # Keep the prefix used to train released checkpoints; its tokens affect inference.
        bbox_prefix_tokens = self._tokenizer.encode("Future interaction ROI: ")

        if bbox is None:
            bbox_tokens: list[int] = []
            bbox_valid = np.zeros(bbox_count, dtype=bool)
        else:
            boxes = np.asarray(bbox, dtype=np.float32)
            if bbox_count == 1 and boxes.shape == (4,):
                boxes = boxes[None]
            if boxes.shape != (bbox_count, 4):
                raise ValueError(f"Expected {bbox_count} future bboxes with shape ({bbox_count}, 4), got {boxes.shape}")
            bbox_valid = np.asarray(bbox_mask, dtype=bool)
            if bbox_count == 1 and bbox_valid.ndim == 0:
                bbox_valid = bbox_valid[None]
            if bbox_valid.shape != (bbox_count,):
                raise ValueError(f"Expected future bbox mask shape ({bbox_count},), got {bbox_valid.shape}")
            location_classes = np.zeros((bbox_count, 4), dtype=np.int32)
            for index in np.flatnonzero(bbox_valid):
                location_classes[index] = future_bbox.source_xyxy_to_location_classes(boxes[index])
            bbox_tokens = future_bbox.location_classes_to_token_ids(location_classes.reshape(-1)).tolist()

        # Reserve the fixed four-token output grammar even at inference, where
        # the targets are absent, so prompt truncation matches training.
        reserved_tokens = len(bbox_prefix_tokens) + 4 * bbox_count
        if reserved_tokens > self._max_len:
            raise ValueError(f"Future bbox branch requires {reserved_tokens} tokens, max length is {self._max_len}")
        if len(prompt_tokens) + reserved_tokens > self._max_len:
            logging.warning(
                "Token length (%d) exceeds max length (%d), truncating the prompt to preserve the future bbox branch.",
                len(prompt_tokens) + reserved_tokens,
                self._max_len,
            )
            prompt_tokens = prompt_tokens[: self._max_len - reserved_tokens]

        tokens = prompt_tokens + bbox_prefix_tokens + bbox_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [False] * len(prompt_tokens) + [False] * len(bbox_prefix_tokens) + [True] * len(bbox_tokens)
        bbox_loss_mask = np.repeat(bbox_valid, 4).tolist() if bbox_tokens else []
        loss_mask = [False] * (len(prompt_tokens) + len(bbox_prefix_tokens)) + bbox_loss_mask

        pad_count = self._max_len - len(tokens)
        tokens += [0] * pad_count
        token_mask += [False] * pad_count
        ar_mask += [False] * pad_count
        loss_mask += [False] * pad_count
        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(token_mask),
            np.asarray(ar_mask, dtype=np.int32),
            np.asarray(loss_mask),
        )

    def _tokenize_prompt(self, prompt: str, state: np.ndarray) -> list[int]:
        """π₀.₅ represents the normalized robot state as discrete language tokens."""
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
        return self._tokenizer.encode(full_prompt, add_bos=True)
