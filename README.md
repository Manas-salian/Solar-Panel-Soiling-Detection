# ☀️ Solar Panel Soiling Detection

An AI-powered computer vision system that automatically detects whether a solar panel is **Clean** or **Dusty / Soiled** from photos using Deep Learning (**MobileNetV2 Transfer Learning**).

---

## 📌 Why This Matters

Dust, sand, and bird droppings accumulate on solar panels over time, reducing solar energy efficiency and power output by up to 25–40%. Automating soiling detection from camera images enables solar farm operators to schedule cleaning only when needed—saving water, reducing maintenance costs, and maximizing clean energy production.

---

## 🧠 Why MobileNetV2 over Standard CNNs?

### 🥊 Head-to-Head Comparison

| Feature | Standard CNN (e.g., ResNet-50) | MobileNet (e.g., MobileNetV2) | Winner |
| :--- | :--- | :--- | :---: |
| **Accuracy** | Higher *(Slightly better at catching complex, granular details)* | Moderate to High *(Very close to standard CNNs, but slightly lower)* | **Standard CNN** |
| **Speed / Latency** | Slower *(Takes longer to process an image)* | Blazing Fast *(Processes images in milliseconds on low-power chips)* | **MobileNet** 🏆 |
| **File Size** | Large *(Can easily be 100MB to 500MB+)* | Tiny *(Usually around 10MB to 15MB)* | **MobileNet** 🏆 |
| **Hardware Required** | Strong Cloud Servers or Dedicated PC GPUs | Smartphones, Raspberry Pi, Edge Devices, Web Browsers | **MobileNet (for flexibility)** 🏆 |

### 💡 Why We Prefer MobileNetV2 for Solar Soiling Detection:
1. **Edge & Drone / Robot Deployment**: Solar installations are often located in remote areas, deserts, or rooftops without high-end GPU servers. MobileNetV2 can run directly on drones, cleaning robots, Raspberry Pis, or embedded IoT microcontrollers.
2. **Real-Time Speed & Low Power Consumption**: Thanks to *Depthwise Separable Convolutions*, MobileNetV2 drastically reduces FLOPs (floating point operations) and battery drain.
3. **Instant Web App Inference**: The small file size (~14 MB) allows instant model loading in web applications (like Streamlit) and mobile apps with near-zero latency.
4. **Optimal Accuracy vs. Resource Trade-off**: For binary classification (Clean vs. Dusty), MobileNetV2 provides high accuracy (~80-90% AUC) without the excessive computational overhead of large 50+ layer networks.

---

## 📂 Project Structure

```text
solar-panel-soiling-detection/
├── app.py                     # Streamlit Web App (Live UI for testing images)
├── main.py                    # Main training & evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── src/                       # Core model & processing code
│   ├── config.py              # Configuration & hyperparameters
│   ├── data_loader.py         # Dataset loading, batching & data augmentation
│   ├── model.py               # MobileNetV2 architecture & fine-tuning logic
│   ├── train.py               # Training routines (frozen backbone + fine-tune)
│   └── evaluate.py            # Metrics calculation & confusion matrix plots
│
├── scripts/                   # Helper utility scripts
│   ├── clean_dataset.py       # Scans & repairs bad/corrupt image files
│   ├── prepare_dataset.py     # Splits raw dataset into train / val / test sets
│   └── predict.py             # CLI tool to predict a single image
│
├── raw_data/                  # Raw images before splitting
│   ├── clean/                 # Clean panel photos
│   └── dusty/                 # Dusty / soiled panel photos
│
├── data/                      # Auto-generated train/val/test splits
│   ├── train/ (clean & dusty)
│   ├── val/   (clean & dusty)
│   └── test/  (clean & dusty)
│
└── outputs/                   # Generated models, metrics, and visualization plots
    ├── solar_soiling_final_model.keras
    ├── metrics.json
    ├── confusion_matrix.png
    └── training_curves_finetune.png
```

---

## 🚀 Quick Start Guide

### Step 1: Environment Setup

1. Open a terminal in the project root directory.
2. Create and activate a Python virtual environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **Mac / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 2: Prepare & Clean Your Dataset

1. Place your raw images inside `raw_data/`:
   - `raw_data/clean/` — photos of clean panels
   - `raw_data/dusty/` — photos of dusty / soiled panels

2. **Clean corrupt files & fix image headers:**
   ```powershell
   python scripts/clean_dataset.py --root raw_data
   ```

3. **Split into Train (70%), Validation (15%), and Test (15%) sets:**
   ```powershell
   Remove-Item -Recurse -Force data\train, data\val, data\test -ErrorAction SilentlyContinue
   python scripts/prepare_dataset.py --raw_dir raw_data --out_dir data
   ```

---

### Step 3: Train the Model

Run the training pipeline:

```powershell
python main.py --data_dir data --epochs 15 --fine_tune_epochs 5 --out_dir outputs
```

**What happens during training:**
- **Stage 1:** Trains classification head with a frozen MobileNetV2 backbone (15 epochs).
- **Stage 2:** Unfreezes the top backbone layers for fine-tuning (5 epochs).
- **Evaluation:** Automatically tests the best checkpoint on the unseen test set and saves performance metrics and plots to `outputs/`.

---

### Step 4: Launch the Web Dashboard

To run the interactive Streamlit dashboard:

```powershell
streamlit run app.py
```

**Features in the Web App:**
- **🔍 Try It Tab:** Drag & drop any solar panel photo to get an instant **Clean vs. Dusty** classification with a confidence score and action recommendation.
- **📊 Training Results Tab:** View loss/accuracy curves, test confusion matrix, and precision/recall metrics.
- **ℹ️ About Tab:** Project background and model specifications.

---

### Step 5: Test a Single Image via Command Line

You can also run quick predictions directly from your terminal on any image:

```powershell
python scripts/predict.py --image "path/to/panel_photo.jpg" --model outputs/solar_soiling_final_model.keras
```

---

## 📊 Model Performance & Artifacts

After training, all artifacts are saved to `outputs/`:
- `solar_soiling_final_model.keras` — Ready-to-deploy trained model.
- `metrics.json` — Detailed precision, recall, and F1-score numbers.
- `confusion_matrix.png` — Visual breakdown of true vs. predicted classes.
- `training_curves_finetune.png` — Accuracy & Loss curves across epochs.

---

## 🎯 Fine-Tuning Logic (How to Adapt It)

**Fine-tuning** is the process of taking a **MobileNetV2** model that has already been trained on a massive dataset (like *ImageNet* with 1.4 million images and 1,000 general categories) and tweaking it to solve your specific task (e.g., detecting clean vs. dusty solar panels).

### 🪜 The 4 Logical Steps for Fine-Tuning MobileNetV2:

1. **Step 1: Freeze the Base Layers**
   - The early layers of the pre-trained MobileNetV2 backbone are frozen so their weights remain fixed.
   - These layers have already mastered fundamental, "general" visual features like edges, curves, gradients, and textures.

2. **Step 2: Replace the Classification Head**
   - The original ImageNet model ends with a dense layer meant to classify 1,000 general object categories.
   - We replace this layer with a custom classification head tailored to our task (`GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1, activation='sigmoid')` for binary Clean vs. Dusty classification).

3. **Step 3: Train the Top Head First**
   - We train only the new custom classification head for a set number of epochs (e.g., 15 epochs) while keeping the backbone frozen.
   - This initializes the new head and prevents large initial gradients from destroying pre-trained feature weights.

4. **Step 4: Unfreeze & Fine-Tune (Top Layers)**
   - We unfreeze the top layers of the MobileNetV2 backbone and retrain the model end-to-end for a few additional epochs (e.g., 5 epochs).
   - A very low learning rate (e.g., `1e-5`) is used to gently adapt higher-level feature detectors to specific dust, grime, and soiling patterns without forgetting general visual features.