# Cloud Anomaly Detection via Convolutional Autoencoders

This project implements an unsupervised anomaly detection pipeline for satellite imagery, distinguishing clear-sky (non-cloud) from cloud-covered scenes by training convolutional autoencoders on non-cloud images and flagging high reconstruction errors as anomalies.


## 📁 Repository Structure

```
cloud-anomaly-detection/
├── scripts/                            # Data loading, model definitions
│   ├── data_utils.py                   # Image preprocessing & augmentation
│   ├── models.py                       # `get_model()` and `get_vgg19()` autoencoder definitions
│   └── train.py                        # Training loop for autoencoder on non-cloud data
├── data/
│   ├── noncloud/                       # 1500 non-cloud images (384×384 RGB)
│   └── cloud/                          # Cloud-covered images for evaluation
├── results/
│   ├── training_logs/                  # Checkpoints and TensorBoard logs
│   └── figures/                        # Sample reconstructions and confusion matrix
└── README.md                           # This file
```


## 🔧 Requirements

* Python 3.7+
* TensorFlow 2.x
* tensorflow-addons
* numpy, pandas, scikit-image, OpenCV
* scikit-learn
* kaggle-datasets
* matplotlib, tqdm

Install via pip:

```bash
pip install tensorflow tensorflow-addons numpy pandas scikit-image opencv-python scikit-learn kaggle-datasets matplotlib tqdm
```


## 🚀 Data Preparation

1. **Dataset paths**: Update `path_noncloud` and `path_cloud` to point to your directories.
2. **Load non-cloud images**: Images are resized to 384 × 384 × 3 and normalized to \[0,1].

   ```python
   for fname in os.listdir(path_noncloud):
       img = keras.preprocessing.image.load_img(..., target_size=(384,384))
       all_images.append(img/255.)
   ```
3. **Train/Test Split**: 80% for autoencoder training, 20% for validation.


## 🛠️ Data Augmentation

Custom augmentations implemented to enrich clear-sky training set:

* Random rotations (±180°)
* Horizontal/vertical flips
* Gaussian noise and blur
* Affine warp shifts

Functions defined in `data_utils.py`:

```python
def anticlockwise_rotation(image): ...

def add_noise(image): ...
# etc.
```


## 🧠 Model Architectures

Two autoencoder variants in `models.py`:

### 1. Convolutional Autoencoder (`get_model()`)

* Encoder: four conv+ReLU blocks (512→256→64→16→8 filters) with max pooling.
* Decoder: symmetric transposed convs reconstruct to 3-channel output.
* Loss: MSE, optimizer: Rectified Adam (RAdam).

### 2. VGG19-based Autoencoder (`get_vgg19()`)

* Encoder: pre-trained VGG19 up to `block5_pool`.
* Decoder: transposed conv layers upsampling back to 384×384×3.
* Trainable end‑to‑end, same MSE/RMSE setup.


## ⚙️ Training

Run `train.py` or the equivalent notebook:

```python
model = get_vgg19()  # or get_model()
model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=100,
    batch_size=16,
    callbacks=[
        ModelCheckpoint('autoencoder.h5', save_best_only=True),
        ReduceLROnPlateau(...), EarlyStopping(...)
    ]
)
```

* **Early stopping** on validation loss (patience=5)
* **Learning rate schedule**: ReduceLROnPlateau (factor 0.4, patience 2)


## 📊 Evaluation

1. **Reconstruction Error**: Compute RMSE per image to separate anomalies.
2. **Thresholding**: Label high-error samples as cloud anomalies.
3. **Metrics**: F1 score, precision, recall, and confusion matrix using ground-truth cloud images.

Example:

```python
preds = model.predict(X_test)
errors = np.sqrt(np.mean((X_test - preds)**2, axis=(1,2,3)))
labels = errors > threshold
print(f1_score(y_true, labels), confusion_matrix(y_true, labels))
```



## ▶️ Usage

1. Adjust paths in `train.py` or your notebook.
2. Install dependencies.
3. Run training script.
4. Evaluate on held-out cloud images and inspect `results/`.

