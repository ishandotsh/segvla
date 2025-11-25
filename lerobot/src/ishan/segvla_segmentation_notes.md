## SegVLA + Mask2Former integration checklist

### What is already wired in code

1. `SegVLAConfig` now exposes segmentation knobs (`segmentation_feature_map`, `segmentation_task_class_ids`, etc.).
2. Policy preprocessing returns `(images, masks, seg_logits)` so VLAFlowMatching knows whether each camera has semantics.
3. `SegmentationCueBuilder` pools Mask2Former logits into global/class/region tokens, builds 32-d geometry hints, and caches a 720-d context vector.
4. `embed_prefix` injects those extra tokens (still 16×768 per camera) before the Connector, and appends the geometry hint via the existing `state_proj` path.
5. `embed_suffix` reuses the action-time MLP path to fuse the 720-d segmentation context with the timestep embedding so the expert transformer is conditioned on semantics.

### Remaining prerequisites before training

1. **Populate `segmentation_feature_map`.** For every camera key in `config.image_features`, provide the batch key that stores its Mask2Former logits (shape `[B, C, H, W]`). Example:
   ```yaml
   segmentation_feature_map:
     observation.images.top: observation.seg_logits_top
     observation.images.wrist: observation.seg_logits_wrist
   ```
2. **Provide segmentation logits.** Either precompute Mask2Former logits when building the dataset or run the model online in the policy preprocessor. See “Prereq 1 playbook” below.
3. **Specify ADE class ids.** Fill `segmentation_task_class_ids` (forces class tokens) and `segmentation_geometry_class_ids` (populates the 32-d geometry vector). Use ADE’s label map, e.g. `bowl=58`, `plate=47`.
4. (Optional) If instructions vary, add a parser that maps nouns in the language prompt to ADE ids per-sample and pass them into the config/batch.

### Prereq 1 playbook – getting Mask2Former logits into the batch

**Option A: precompute during dataset creation (recommended)**

1. Install Mask2Former deps (`pip install transformers torchvision timm`).
2. Load `facebook/mask2former-swin-large-ade-semantic` and iterate over your existing dataset.
3. For each RGB frame (per camera):
   - Apply the same resize/pad pipeline used in `prepare_images` (512×512 with padding).
   - Convert to the pixel range expected by Mask2Former (usually `[0,1]`).
   - Run the model, grab the raw class logits `P ∈ ℝ^{C×H×W}` before softmax.
   - Save `P` (float16/32) alongside the frame under a key like `observation.seg_logits_top`.
4. Update your dataset schema so these tensors are available when training/inference.

**Option B: compute online in the policy processor**

1. Add a new `ProcessorStep` (e.g., `SegVLAAddSegmentationStep`) that:
   - Receives RGB tensors from the batch.
   - Resizes/pads, runs Mask2Former, stores logits under the keys referenced by `segmentation_feature_map`.
2. Insert this step before normalization/tokenization in `make_segvla_pre_post_processors`.
3. Beware of latency: Mask2Former per frame is heavy, so this is mainly for prototyping.

### Tracking work ownership

| Task | Owner |
| --- | --- |
| Choose ADE class ids and set config fields | Ishan |
| Produce/pipe Mask2Former logits (Option A or B) | Ishan (we can help with scripts if needed) |
| Fine-tune SegVLA with new cues | Ishan |
| Future tweaks (e.g., per-task noun mapping) | both |

