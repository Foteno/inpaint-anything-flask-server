from flask import Flask, request, jsonify
from resource_management import preload_sam, predict_sam
from utils import save_array_to_img, load_img_to_array, dilate_mask
import time
from PIL import Image
from pathlib import Path
import numpy as np
import ipdb
from lama_inpaint import inpaint_img_with_lama
import cv2

app = Flask(__name__)
images = {}
sam_predictor = None
small_prefix = "small_"
dilated_prefix = "dilated_"

out_dir = Path("./temporary")
out_dir.mkdir(parents=True, exist_ok=True)
image_path = "image.png"

lama_config = "./lama/configs/prediction/default.yaml"
lama_ckpt = "./pretrained_models/big-lama"

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/post-point-sam', methods=['POST'])
def post_point_sam():
    
    image_filenames = []
    image_file = request.files.get('image')
    coord_x = request.args['coord_x']
    coord_y = request.args['coord_y']

    # Read the image file and perform processing
    image = load_img_to_array(image_file)
    save_array_to_img(image, out_dir / "image.png")

    sam_predictor.set_image(image)
    start_time = time.perf_counter()
    
    masks, scores, logits = predict_sam(sam_predictor, float(coord_x), float(coord_y))

    end_time = time.perf_counter()
    latency = (end_time - start_time) 
    print("Segmentation latency: %.5f seconds" % latency)

    dilate_seize = 15
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        mask_preview_p = out_dir / f"{small_prefix}mask_{idx}.png"
        mask_dilated_p = out_dir / f"{dilated_prefix}mask_{idx}.png"

        save_array_to_img(mask, mask_p)
        save_array_to_img(cv2.resize(mask, None, fx=0.3, fy=0.3), mask_preview_p)
        save_array_to_img(dilate_mask(mask, dilate_seize), mask_dilated_p)
        # images.put(idx, mask_p) #add cleanup
        image_filenames.append(f"mask_{idx}.png")

    response = jsonify({'message': 'Image processed successfully', 'masks': image_filenames})
    return response, 200

@app.route('/get-mask', methods=['GET'])
def get_mask():
    mask_file_value = request.args['mask_file']
    mask_file_value = small_prefix + mask_file_value
    with open(out_dir / mask_file_value, 'rb') as f:
        image_file = f.read()
    
    return image_file

@app.route('/choose-mask', methods=['GET'])
def choose_mask():
    mask_name = request.args['mask_file']
    mask_path = out_dir / mask_name

    mask = load_img_to_array(mask_path)
    mask = mask.astype(np.uint8) * 255
    
    dilate_size = request.args.get('dilate')
    if dilate_size is not None:
        print(f"Starting dilation of {mask_name}")
        mask = dilate_mask(mask, int(dilate_size))

    start_time = time.perf_counter()
    print(f"Starting inpainting of {mask_name}")
    image = load_img_to_array(out_dir / image_path)
    img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_path).name}"
    img_inpainted = inpaint_img_with_lama(image, mask, lama_config, lama_ckpt, device="cpu")
    save_array_to_img(img_inpainted, img_inpainted_p)

    end_time = time.perf_counter()
    latency = (end_time - start_time) 
    print("Inpainting latency: %.5f seconds" % latency)
    
    with open(img_inpainted_p, 'rb') as f:
        image_file = f.read()
    
    return image_file

if __name__ == '__main__':
    print("Loading SAM model...")
    sam_predictor = preload_sam()
    print("Preloaded SAM model") 
    app.run(debug=True, host='localhost', port=5000)