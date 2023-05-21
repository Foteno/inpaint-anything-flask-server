# Purpose: To manage the resources for the project
from segment_anything import SamPredictor, sam_model_registry
import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_tensor_to_modulo

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

def preload_lama(config_p: str, ckpt_p, device="cuda"):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    return model