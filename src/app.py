import os
import io
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from PIL import Image
import rasterio
from skimage.transform import resize as sk_resize
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50.keras"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GT_FOLDER = BASE_DIR / "static" / "gt_masks"  
ALLOWED_EXT = {".tif", ".tiff"}

UPLOAD_FOLDER.mkdir(exist_ok=True)
GT_FOLDER.mkdir(exist_ok=True)

channel_names = [
    "Coastal aerosol","Blue","Green","Red","NIR","SWIR1","SWIR2","QA Band",
    "Merit DEM","Copernicus DEM","ESA World Cover map","Water occurrence probability"
]

best_channels = ["Red", "Green", "Merit DEM"]
best_channels_idx = [channel_names.index(ch) for ch in best_channels]

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)

print("🔹 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss})
    print("✅ Model loaded with custom objects.")
except Exception as e:
    print("⚠️ Could not load with custom objects, trying simple load...", e)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")

preprocess_fn = tf.keras.applications.resnet.preprocess_input

def read_tif(filepath):
    with rasterio.open(filepath) as src:
        arr = src.read()  # (C,H,W)
        arr = np.transpose(arr, (1,2,0))  # -> (H,W,C)
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr

def select_and_resize(img, channel_indices, target_size=(128,128)):
    valid_idx = [i for i in channel_indices if i < img.shape[-1]]
    selected = img[:,:,valid_idx]
    resized = sk_resize(selected, target_size, preserve_range=True, anti_aliasing=True)
    return resized.astype(np.float32)

def prepare_for_model(img_resized):
    img = img_resized * 255.0
    img = preprocess_fn(img)
    return img

def predict_mask(img_array):
    inp = np.expand_dims(img_array, axis=0)
    pred = model.predict(inp)
    pred = np.squeeze(pred)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[:,:,0]
    return pred

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def mask_to_overlay(rgb_image, mask_prob, alpha=0.5):
    rgb = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-6)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    mask_uint8 = (mask_prob * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb_uint8)
    mask_pil = Image.fromarray(mask_uint8).convert("L")
    mask_rgba = Image.new("RGBA", rgb_pil.size)
    mask_rgba.paste((255,0,0,int(255*alpha)), mask=mask_pil)
    base = rgb_pil.convert("RGBA")
    out = Image.alpha_composite(base, mask_rgba)
    return out

def image_to_base64(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b}"

app = Flask(__name__)
app.secret_key = "dev-key"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", channel_names=channel_names, best_channels=best_channels_idx)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("index"))
    f = request.files["file"]
    if f.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    filename = f.filename
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        flash("Only .tif or .tiff allowed")
        return redirect(url_for("index"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{Path(filename).stem}_{timestamp}{ext}"
    save_path = UPLOAD_FOLDER / saved_name
    f.save(save_path)

    selected = request.form.getlist("channels")
    selected_idx = [int(s) for s in selected] if selected else best_channels_idx

    try:
        img = read_tif(str(save_path))
        resized = select_and_resize(img, selected_idx)
        prep = prepare_for_model(resized)
        pred_prob = predict_mask(prep)
        pred_bin = (pred_prob > 0.5).astype(np.uint8)

        vis_rgb = resized[:,:,:3] if resized.shape[-1] >= 3 else np.repeat(resized[:,:,0:1], 3, axis=-1)
        overlay_pil = mask_to_overlay(vis_rgb, pred_prob)
        mask_pil = Image.fromarray((pred_bin*255).astype(np.uint8)).convert("L")

        out_prefix = GT_FOLDER / Path(saved_name).stem
        mask_pil.save(f"{out_prefix}_pred.png")

        gt_mask_path = GT_FOLDER / f"{Path(filename).stem}_gt.png"
        iou_value = None
        if gt_mask_path.exists():
            gt_mask = np.array(Image.open(gt_mask_path).convert("L")) > 128
            iou_value = calculate_iou(pred_bin, gt_mask)

        return render_template(
            "index.html",
            channel_names=channel_names,
            best_channels=best_channels_idx,
            uploaded_filename=saved_name,
            overlay_img=image_to_base64(overlay_pil),
            mask_img=image_to_base64(mask_pil),
            preview_img=image_to_base64(Image.fromarray((vis_rgb*255).astype(np.uint8))),
            iou_value=iou_value
        )

    except Exception as e:
        flash(f"Error processing image: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
