#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.segvla.configuration_segvla import SegVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SegVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SegVLAConfig
    name = "segvla"

    def __init__(
        self,
        config: SegVLAConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks, seg_logits = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, seg_logits, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks, seg_logits = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(
            images, img_masks, seg_logits, lang_tokens, lang_masks, state, actions, noise, time
        )
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        seg_logits = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device

            # Padding masks can come from datasets as "<key>_padding_mask" or "<key>_is_pad" (True = padded).
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"]
            elif f"{key}_is_pad" in batch:
                mask = ~batch[f"{key}_is_pad"]  # convert is_pad → valid-token mask
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            mask = mask.to(device=device, dtype=torch.bool)

            # If the mask still has a time/frame dimension, mirror the image selection (last frame).
            if mask.ndim == 1 and mask.shape[0] != bsize:
                mask = mask.view(1, -1).expand(bsize, -1)
            if mask.ndim > 1:
                mask = mask[:, -1]

            if mask.shape[0] != bsize:
                raise ValueError(
                    f"Unexpected padding mask shape {mask.shape} for {key}; expected batch dimension {bsize}"
                )
            images.append(img)
            img_masks.append(mask)

            seg_key = self.config.segmentation_feature_map.get(key)
            if seg_key is not None and seg_key in batch:
                seg = batch[seg_key][:, -1, :, :, :] if batch[seg_key].ndim == 5 else batch[seg_key]
                seg_logits.append(seg)
            else:
                seg_logits.append(None)
                print("seg_logits is none")

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
            seg_logits.append(None)
        return images, img_masks, seg_logits

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


@dataclass
class SegmentationCueResult:
    tokens: torch.Tensor
    class_ids: torch.Tensor
    presence: torch.Tensor
    centroids: torch.Tensor
    areas: torch.Tensor

    def gather_stats(self, class_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.tokens.device
        dtype = self.tokens.dtype
        if self.class_ids.numel() == 0:
            zeros = torch.zeros(self.tokens.shape[0], dtype=dtype, device=device)
            centroid = torch.zeros(self.tokens.shape[0], 2, dtype=dtype, device=device)
            return zeros, centroid, zeros
        mask = self.class_ids == class_id
        if not torch.any(mask):
            zeros = torch.zeros(self.tokens.shape[0], dtype=dtype, device=device)
            centroid = torch.zeros(self.tokens.shape[0], 2, dtype=dtype, device=device)
            return zeros, centroid, zeros
        presence = torch.where(mask, self.presence, torch.zeros_like(self.presence)).sum(dim=1)
        centroid = torch.where(mask[..., None], self.centroids, torch.zeros_like(self.centroids)).sum(dim=1)
        areas = torch.where(mask, self.areas, torch.zeros_like(self.areas)).sum(dim=1)
        return presence, centroid, areas


class SegmentationCueBuilder(nn.Module):
    def __init__(
        self,
        patch_shape: tuple[int, int],
        hidden_dim: int,
        num_classes: int,
        max_tokens: int,
        region_specs: tuple[tuple[float, float, float, float], ...],
        top_k: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_tokens = max_tokens
        self.region_specs = region_specs
        self.top_k = top_k
        self.dropout_prob = dropout_prob
        self.num_region_tokens = len(region_specs)
        self.num_class_slots = max(0, max_tokens - 1 - self.num_region_tokens)

        patch_h, patch_w = patch_shape
        y_coords = torch.linspace(0.0, 1.0, patch_h)
        x_coords = torch.linspace(0.0, 1.0, patch_w)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.register_buffer("coord_x", grid_x.reshape(-1), persistent=False)
        self.register_buffer("coord_y", grid_y.reshape(-1), persistent=False)

        region_masks = []
        for (x0, x1, y0, y1) in region_specs:
            mask = torch.zeros(patch_h, patch_w)
            xs = slice(int(x0 * patch_w), max(int(x1 * patch_w), int(x0 * patch_w) + 1))
            ys = slice(int(y0 * patch_h), max(int(y1 * patch_h), int(y0 * patch_h) + 1))
            mask[ys, xs] = 1.0
            region_masks.append(mask.reshape(-1))
        if region_masks:
            self.register_buffer("region_masks", torch.stack(region_masks), persistent=False)
        else:
            self.register_buffer("region_masks", torch.empty(0, patch_h * patch_w), persistent=False)

        self.null_tokens = nn.Parameter(torch.zeros(max_tokens, hidden_dim))

    def forward(
        self,
        vision_tokens: torch.Tensor,
        seg_logits: torch.Tensor | None,
        class_ids: list[int] | None = None,
    ) -> SegmentationCueResult | None:
        if seg_logits is None:
            return None
        if self.training and self.dropout_prob > 0.0:
            if torch.rand(1, device=vision_tokens.device).item() < self.dropout_prob:
                return None

        seg = seg_logits.to(device=vision_tokens.device, dtype=vision_tokens.dtype)
        seg = F.interpolate(seg, size=self.patch_shape, mode="bilinear", align_corners=False)
        seg_probs = seg.softmax(dim=1)
        bsz, _, _, _ = seg_probs.shape
        seg_probs = seg_probs.flatten(2)

        top_k = min(self.top_k, self.num_classes)
        mean_probs = seg_probs.mean(dim=-1)
        topk_vals, topk_idx = mean_probs.topk(k=top_k, dim=1)

        class_tokens = torch.zeros(
            bsz, self.num_class_slots, self.hidden_dim, dtype=vision_tokens.dtype, device=vision_tokens.device
        )
        selected_ids = torch.full(
            (bsz, self.num_class_slots), -1, dtype=torch.long, device=vision_tokens.device
        )
        presence = torch.zeros(bsz, self.num_class_slots, dtype=vision_tokens.dtype, device=vision_tokens.device)
        centroids = torch.zeros(bsz, self.num_class_slots, 2, dtype=vision_tokens.dtype, device=vision_tokens.device)
        areas = torch.zeros_like(presence)

        additional_ids = class_ids or []
        for b in range(bsz):
            chosen = topk_idx[b].tolist()
            for cid in additional_ids:
                if 0 <= cid < self.num_classes and cid not in chosen:
                    chosen.append(cid)
            chosen = chosen[: self.num_class_slots]
            for slot, class_id in enumerate(chosen):
                weights = seg_probs[b, class_id]
                alpha = torch.softmax(weights, dim=0)
                token = torch.sum(alpha[:, None] * vision_tokens[b], dim=0)
                total = weights.sum() + 1e-6
                cx = torch.sum(weights * self.coord_x) / total
                cy = torch.sum(weights * self.coord_y) / total
                class_tokens[b, slot] = token
                selected_ids[b, slot] = class_id
                presence[b, slot] = weights.mean()
                centroids[b, slot, 0] = cx
                centroids[b, slot, 1] = cy
                areas[b, slot] = weights.mean()

        global_token = vision_tokens.mean(dim=1, keepdim=True)
        region_tokens = []
        for mask in self.region_masks:
            norm = mask.sum()
            if norm <= 0:
                continue
            weights = mask / norm
            token = torch.sum(vision_tokens * weights[None, :, None], dim=1, keepdim=True)
            region_tokens.append(token)
        if region_tokens:
            region_tokens = torch.cat(region_tokens, dim=1)
        else:
            region_tokens = torch.zeros(
                bsz, 0, self.hidden_dim, dtype=vision_tokens.dtype, device=vision_tokens.device
            )

        token_parts = [global_token, class_tokens]
        if isinstance(region_tokens, torch.Tensor) and region_tokens.numel() > 0:
            token_parts.append(region_tokens)
        tokens = torch.cat(token_parts, dim=1)
        if tokens.shape[1] < self.max_tokens:
            pad = self.null_tokens[: self.max_tokens - tokens.shape[1]][None, :, :].to(tokens.dtype)
            tokens = torch.cat([tokens, pad.expand(bsz, -1, -1)], dim=1)
        elif tokens.shape[1] > self.max_tokens:
            tokens = tokens[:, : self.max_tokens]

        return SegmentationCueResult(tokens=tokens, class_ids=selected_ids, presence=presence, centroids=centroids, areas=areas)

class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SegVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        if self.config.segmentation_feature_map:
            self.segmentation_builder = SegmentationCueBuilder(
                patch_shape=self.vlm_with_expert.vision_patch_shape,
                hidden_dim=self.vlm_with_expert.vision_hidden_size,
                num_classes=self.config.segmentation_num_classes,
                max_tokens=self.config.segmentation_max_tokens,
                region_specs=self.config.segmentation_region_specs,
                top_k=self.config.segmentation_top_k_classes,
                dropout_prob=self.config.segmentation_dropout_prob,
            )
            text_hidden = self.vlm_with_expert.config.text_config.hidden_size
            expert_hidden = self.vlm_with_expert.expert_hidden_size
            self.segmentation_token_proj = nn.Linear(self.vlm_with_expert.vision_hidden_size, text_hidden)
            self.segmentation_state_proj = nn.Linear(self.config.segmentation_geometry_state_dim, text_hidden)
            self.segmentation_context_proj = nn.Linear(self.vlm_with_expert.vision_hidden_size, expert_hidden)
            self.segmentation_time_fuser = nn.Sequential(
                nn.Linear(expert_hidden * 2, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, expert_hidden),
                nn.LayerNorm(expert_hidden),
            )
        else:
            self.segmentation_builder = None
            self.segmentation_token_proj = None
            self.segmentation_state_proj = None
            self.segmentation_context_proj = None
            self.segmentation_time_fuser = None

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, seg_logits, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        target_bsize = images[0].shape[0]

        embs: list[torch.Tensor] = []
        emb_names: list[str] = []
        pad_masks = []
        att_masks = []
        seg_context_tokens = []
        geometry_candidates: list[torch.Tensor] = []

        for _img_idx, (
            img,
            img_mask,
            seg_logit,
        ) in enumerate(zip(images, img_masks, seg_logits, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                emb_names.append(f"{key}_img_start")
                pad_masks.append(image_start_mask)

            img_emb, patch_tokens = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask.to(device=img_emb.device, dtype=torch.bool)

            # Squeeze/reshape any extra dims (e.g., [B, T], [T], [T, 1])
            if img_mask.ndim > 1:
                img_mask = img_mask.reshape(img_mask.shape[0], -1)
                if img_mask.shape[0] == bsize:
                    img_mask = img_mask[:, -1]
                else:
                    img_mask = img_mask.flatten()

            # Align to batch dimension, broadcasting if mask is frame-only.
            if img_mask.ndim == 1 and img_mask.shape[0] != bsize:
                if img_mask.numel() == 1:
                    img_mask = img_mask.expand(bsize)
                elif img_mask.shape[0] < bsize:
                    img_mask = img_mask.view(1, -1).expand(bsize, -1)[:, -1]
                else:
                    img_mask = img_mask[:bsize]

            if img_mask.shape[0] != bsize:
                raise ValueError(f"Image mask shape {img_mask.shape} does not match batch size {bsize}.")

            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            emb_names.append(f"{key}_img")
            pad_masks.append(img_mask)
 
            att_masks += [0] * (num_img_embs)
            if self.segmentation_builder is not None and seg_logit is not None:
                seg_result = self.segmentation_builder(patch_tokens, seg_logit, self.config.segmentation_task_class_ids)
                if seg_result is not None:
                    seg_tokens = self.segmentation_token_proj(seg_result.tokens)
                    seg_mask = img_mask[:, :1].expand(bsize, seg_tokens.shape[1])
                    embs.append(seg_tokens)
                    emb_names.append(f"{key}_seg")
                    pad_masks.append(seg_mask)
                    att_masks += [0] * seg_tokens.shape[1]
                    seg_context_tokens.append(seg_result.tokens)
                    geom = self._build_geometry_features(seg_result)
                    if geom is not None:
                        geometry_candidates.append(geom)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                emb_names.append(f"{key}_img_end")
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        # Normalize language tensors to match the image batch size.
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens[None, :]
        if lang_masks.ndim == 1:
            lang_masks = lang_masks[None, :]
        if lang_tokens.shape[0] != target_bsize or lang_masks.shape[0] != target_bsize:
            if lang_tokens.shape[0] == 1 and lang_masks.shape[0] == 1:
                lang_tokens = lang_tokens.expand(target_bsize, -1)
                lang_masks = lang_masks.expand(target_bsize, -1)
            else:
                raise ValueError(
                    f"Language batch size mismatch: tokens {lang_tokens.shape}, masks {lang_masks.shape}, "
                    f"expected {target_bsize} (image batch)."
                )

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        emb_names.append("lang")
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        emb_names.append("state")
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)

        geometry = self._aggregate_geometry_vectors(
            geometry_candidates, device=state_emb.device, dtype=state_emb.dtype
        )
        if geometry is not None and self.segmentation_state_proj is not None:
            geom_emb = self.segmentation_state_proj(geometry)
            geom_emb = geom_emb[:, None, :]
            geom_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            embs.append(geom_emb)
            emb_names.append("geom")
            pad_masks.append(geom_mask)
            att_masks += [1]

        # Ensure all embeddings are 3D (B, T, D) before concatenation.
        bsize_ref = None
        for idx, emb in enumerate(embs):
            if emb.ndim == 2:
                emb = emb[:, None, :]
            elif emb.ndim != 3:
                raise ValueError(
                    f"Unexpected embedding rank {emb.ndim} at index {idx} ({emb_names[idx]}), shape {emb.shape}"
                )

            if bsize_ref is None:
                bsize_ref = emb.shape[0]
            elif emb.shape[0] != bsize_ref:
                shapes = [(name, e.shape) for name, e in zip(emb_names, embs)]
                raise ValueError(
                    f"Embedding batch mismatch: expected {bsize_ref}, got {emb.shape[0]} "
                    f"at index {idx} ({emb_names[idx]}). All shapes: {shapes}"
                )
            embs[idx] = emb

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        seg_context = torch.cat(seg_context_tokens, dim=1) if seg_context_tokens else None
        seg_metadata = {"context_tokens": seg_context, "geometry": geometry}

        return embs, pad_masks, att_masks, seg_metadata

    def _build_geometry_features(self, seg_result: SegmentationCueResult) -> torch.Tensor | None:
        if not self.config.segmentation_geometry_class_ids:
            return None
        device = seg_result.tokens.device
        dtype = seg_result.tokens.dtype
        geom_dim = self.config.segmentation_geometry_state_dim
        geom = torch.zeros(seg_result.tokens.shape[0], geom_dim, device=device, dtype=dtype)
        offset = 0
        centroids = []
        for class_id in self.config.segmentation_geometry_class_ids:
            presence, centroid, area = seg_result.gather_stats(class_id)
            values = torch.stack([presence, centroid[:, 0], centroid[:, 1], area], dim=1)
            end = min(offset + 4, geom_dim)
            geom[:, offset:end] = values[:, : end - offset]
            offset = end
            centroids.append(centroid)
            if offset >= geom_dim:
                break
        if len(centroids) >= 2 and offset + 5 <= geom_dim:
            dx = centroids[0][:, 0] - centroids[1][:, 0]
            dy = centroids[0][:, 1] - centroids[1][:, 1]
            dist = torch.sqrt(dx**2 + dy**2 + 1e-6)
            angle = torch.atan2(dy, dx)
            rel = torch.stack([dx, dy, dist, torch.sin(angle), torch.cos(angle)], dim=1)
            geom[:, offset : offset + rel.shape[1]] = rel[:, : geom_dim - offset]
        return geom

    def _aggregate_geometry_vectors(
        self, geometry_candidates: list[torch.Tensor], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        valid = [g for g in geometry_candidates if g is not None]
        if not valid:
            return None
        stacked = torch.stack(valid, dim=0)
        return stacked.mean(dim=0).to(dtype=dtype, device=device)

    def embed_suffix(self, noisy_actions, timestep, seg_context: torch.Tensor | None = None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        if (
            seg_context is not None
            and self.segmentation_context_proj is not None
            and self.segmentation_time_fuser is not None
        ):
            seg_vec = seg_context.mean(dim=1)
            seg_vec = self.segmentation_context_proj(seg_vec)
            if self.training and self.config.segmentation_dropout_prob > 0:
                drop_mask = torch.rand(seg_vec.shape[0], device=device) < self.config.segmentation_dropout_prob
                seg_vec = torch.where(drop_mask[:, None], torch.zeros_like(seg_vec), seg_vec)
            seg_vec = seg_vec[:, None, :].expand_as(action_emb)
            fused = torch.cat([time_emb, seg_vec], dim=2)
            time_emb = self.segmentation_time_fuser(fused)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, seg_logits, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks, seg_metadata = self.embed_prefix(
            images, img_masks, seg_logits, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, time, seg_metadata.get("context_tokens")
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, seg_logits, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            seg_metadata,
        ) = self.embed_prefix(images, img_masks, seg_logits, lang_tokens, lang_masks, state=state)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        seg_context = seg_metadata.get("context_tokens") if isinstance(seg_metadata, dict) else None
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                seg_context,
            )
            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        seg_context: torch.Tensor | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep, seg_context)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
