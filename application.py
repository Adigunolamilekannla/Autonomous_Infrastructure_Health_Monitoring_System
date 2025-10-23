import os
import pandas as pd
import numpy as np
import joblib
import torch
from PIL import Image
from flask import Flask, request, render_template
from src.utils.model import LSTMNet, get_bridge_img_model_optimizer

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ----------------------------
# Load all models
# ----------------------------
device = torch.device("cpu")

def load_joblib(path):
    return joblib.load(path) if os.path.exists(path) else None

def load_torch_model(model, path):
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
    return model

# Paths
RUL_MODEL_PATH = "artifacts/models/rul_model.joblib"
SENSOR_MODEL_PATH = "artifacts/models/bridge_model.pt"
IMAGE_MODEL_PATH = "artifacts/models/bridge_img_model.pt"
FUSION_MODEL_PATH = "artifacts/models/fulsion_model.joblib"
SCALER_PATH = "artifacts/models/scaler_model.joblib"

# Load models
rul_model = load_joblib(RUL_MODEL_PATH)
fusion_model = load_joblib(FUSION_MODEL_PATH)
scaler = load_joblib(SCALER_PATH)

# LSTM for sensor
lstm_model = LSTMNet(23, 20, 10)
lstm_model = load_torch_model(lstm_model, SENSOR_MODEL_PATH).to(device)

# CNN for image
cnn_model = get_bridge_img_model_optimizer()[0]
cnn_model = load_torch_model(cnn_model, IMAGE_MODEL_PATH).to(device)

# Store intermediate predictions
rul_preds, sensor_preds, image_preds = None, None, None

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_rul", methods=["POST"])
def predict_rul():
    global rul_preds
    try:
        file = request.files["file"]
        data = pd.read_csv(file)
        if "RUL" in data.columns:
            data = data.drop(columns=["RUL"])
        data = np.array(data)
        rul_preds = rul_model.predict(data)
        return render_template("result.html", predictions=rul_preds.tolist())
    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/predict_sensor", methods=["POST"])
def predict_sensor():
    global sensor_preds
    try:
        file = request.files["file"]
        data = pd.read_csv(file)
        if "Unnamed: 0" in data.columns:
            data = data.drop("Unnamed: 0", axis=1)
        data = scaler.transform(data)
        X = torch.tensor(data, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs, _ = lstm_model(X)
        sensor_preds = np.array((outputs >= 0.5).float()).squeeze()
        sensor_prediction = "Bridge is Standing" if sensor_preds == 1 else "Bridge is Collapsing"
        return render_template("result.html", predictions=[sensor_prediction])
    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/predict_image", methods=["POST"])
def predict_image():
    global image_preds
    try:
        file = request.files["file"]

        # save uploaded file
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)

        # open and preprocess image
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # make prediction
        with torch.no_grad():
            output = cnn_model(img_tensor)
            image_preds = (output >= 0.5).float().cpu().numpy().flatten()

        if image_preds is not None and len(image_preds) > 0:
            return render_template("result.html", predictions=image_preds.tolist())
        else:
            return render_template("result.html", error="Model returned no predictions.")
            
    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    global rul_preds, sensor_preds, image_preds
    try:
        if any(x is None for x in [rul_preds, sensor_preds, image_preds]):
            raise ValueError("Please run RUL, Sensor, and Image predictions first.")

        # Combine features
        features = np.hstack([
            np.mean(rul_preds),
            np.mean(sensor_preds),
            np.mean(image_preds)
        ]).reshape(1, -1)

        # Predict
        fusion_pred = fusion_model.predict(features)[0]

        # Convert numeric to text
        if fusion_pred == 0:
            fusion_pred = "Based On Our Model Final Report The Bridged Condiction is very Bad"
        elif fusion_pred == 1:
            fusion_pred = "Based On Our Model Final Report The Bridged Condiction is Moderate"
        else:
            fusion_pred = "Based On Our Model Final Report The Bridged Condiction is Good"

        return render_template("result.html", predictions=[fusion_pred])

    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

