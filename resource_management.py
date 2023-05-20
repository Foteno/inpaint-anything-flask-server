# Purpose: To manage the resources for the project
from segment_anything import SamPredictor, sam_model_registry
import numpy as np

# model = "segment_anything"
# model_type="vit_b"
# ckpt_p="./pretrained_models/sam_vit_b_01ec64.pth"

def preload_sam(device="cuda", model_type="vit_b", ckpt_p="./pretrained_models/sam_vit_b.pth"):
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def predict_sam(samPredictor, coord_x, coord_y):
    coords = np.array([[coord_x, coord_y]])
    print(coords)
    print(type(coords))
    masks, scores, logits = samPredictor.predict(
        point_coords=coords,
        point_labels=[1],
        multimask_output=True,
    )
    masks = masks.astype(np.uint8) * 255
    return masks, scores, logits