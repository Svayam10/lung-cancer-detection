# ================================================================
# LUNG CANCER DETECTION: U-NET (Baseline vs Attention) + CNN
# Complete Implementation for VSCode - FIXED VERSION
# ================================================================

"""
PROJECT STRUCTURE:
lung-cancer-detection/
├── main.py (this file)
├── data/
│   ├── CXR_png/          # Your CT images
│   ├── masks/            # Your mask images
│   └── ClinicalReadings/ # Your patient data
├── models/
│   ├── unet_baseline.h5
│   ├── unet_attention.h5
│   └── cnn_classifier.h5
└── results/
    ├── segmentation_baseline/
    ├── segmentation_attention/
    ├── classification_results/
    └── metrics_comparison.png

INSTRUCTIONS:
1. Create the folder structure above
2. Place your datasets in the data/ folder
3. Run sections one by one (they're marked)
4. Results will be saved in results/ folder
"""

# ================================================================
# SECTION 1: IMPORTS AND SETUP
# ================================================================
print("="*60)
print("SECTION 1: IMPORTS AND SETUP")
print("="*60)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, Multiply, Add, Activation, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/segmentation_baseline', exist_ok=True)
os.makedirs('results/segmentation_attention', exist_ok=True)
os.makedirs('results/classification_results', exist_ok=True)

print("✓ Setup complete!")
print()

# ================================================================
# SECTION 2: DATA LOADING FUNCTIONS
# ================================================================
print("="*60)
print("SECTION 2: DATA LOADING")
print("="*60)

# Update these paths to your actual data location
CT_IMAGES_PATH = 'data/CXR_png'
MASKS_PATH = 'data/masks'
CLINICAL_PATH = 'data/ClinicalReadings'
IMG_SIZE = (256, 256)

def load_images_and_masks(ct_images_path, masks_path, img_size):
    """Load ONLY CT images that have corresponding masks"""
    ct_images = []
    masks = []
    filenames = []
    
    # Collect all available mask filenames
    available_masks = {}
    for filename in os.listdir(masks_path):
        if filename.endswith('_mask.png'):
            original_name = filename.replace('_mask.png', '.png')
            available_masks[original_name] = filename
    
    print(f"Found {len(available_masks)} masks in masks folder")
    
    # Load ALL CT images
    ct_files = [f for f in os.listdir(ct_images_path) if f.endswith('.png')]
    print(f"Found {len(ct_files)} CT images")
    
    skipped = 0
    loaded = 0
    
    for filename in ct_files:
        # Skip if no mask available
        if filename not in available_masks:
            skipped += 1
            continue
        
        image_path = os.path.join(ct_images_path, filename)
        mask_path = os.path.join(masks_path, available_masks[filename])
        
        # Read CT image
        ct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if ct_image is not None and mask is not None:
            ct_image = cv2.resize(ct_image, img_size)
            ct_image = ct_image.astype(np.float32) / 255.0
            
            mask = cv2.resize(mask, img_size)
            mask = mask.astype(np.float32) / 255.0
            
            ct_images.append(ct_image)
            masks.append(mask)
            filenames.append(filename)
            loaded += 1
        else:
            skipped += 1
    
    print(f"\n✓ Successfully loaded {loaded} image-mask pairs")
    print(f"✗ Skipped {skipped} images (no mask or corrupted)")
    
    return np.array(ct_images), np.array(masks), filenames

# Load data
ct_images, masks, filenames = load_images_and_masks(CT_IMAGES_PATH, MASKS_PATH, IMG_SIZE)

# Reshape for model input
ct_images = ct_images.reshape(-1, 256, 256, 1)
masks = masks.reshape(-1, 256, 256, 1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    ct_images, masks, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print()

# ================================================================
# SECTION 3: MODEL ARCHITECTURES
# ================================================================
print("="*60)
print("SECTION 3: MODEL ARCHITECTURES")
print("="*60)

# ----- BASELINE U-NET (No Attention) -----
def unet_baseline(input_size=(256, 256, 1)):
    """Standard U-Net without attention"""
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# ----- U-NET WITH ATTENTION GATES -----
def attention_gate(x, g, inter_channels):
    """Attention Gate Implementation"""
    # Theta path (x)
    theta_x = Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    
    # Phi path (g)
    phi_g = Conv2D(inter_channels, 1, strides=1, padding='same')(g)
    
    # Add and apply ReLU
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    
    # Psi
    psi = Conv2D(1, 1, strides=1, padding='same')(act_xg)
    psi = Activation('sigmoid')(psi)
    
    # Multiply attention coefficients with input
    upsample_psi = UpSampling2D(size=(1, 1))(psi)
    y = Multiply()([x, upsample_psi])
    
    return y

def unet_attention(input_size=(256, 256, 1)):
    """U-Net with Attention Gates"""
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder with Attention Gates
    up1 = UpSampling2D(size=(2, 2))(conv3)
    att1 = attention_gate(conv2, up1, 128)  # Attention gate
    merge1 = concatenate([att1, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    att2 = attention_gate(conv1, up2, 64)  # Attention gate
    merge2 = concatenate([att2, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

print("✓ Model architectures defined")
print()

# ================================================================
# SECTION 4: TRAIN BASELINE U-NET
# ================================================================
print("="*60)
print("SECTION 4: BASELINE U-NET")
print("="*60)

baseline_model_path = 'models/unet_baseline.h5'

# Check if model already exists
if os.path.exists(baseline_model_path):
    print(f"✓ Found existing model: {baseline_model_path}")
    response = input("Do you want to use the existing model? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("Loading existing baseline model...")
        model_baseline = tf.keras.models.load_model(baseline_model_path)
        print("✓ Baseline model loaded successfully")
        
        # Evaluate to get history-like metrics
        print("Evaluating model on validation set...")
        val_results = model_baseline.evaluate(X_val, y_val, verbose=0)
        
        # Create a mock history object for plotting
        history_baseline = type('obj', (object,), {
            'history': {
                'accuracy': [val_results[1]] * 10,
                'val_accuracy': [val_results[1]] * 10,
                'loss': [val_results[0]] * 10,
                'val_loss': [val_results[0]] * 10,
                'precision': [val_results[2]] * 10,
                'val_precision': [val_results[2]] * 10,
                'recall': [val_results[3]] * 10,
                'val_recall': [val_results[3]] * 10
            }
        })()
        print(f"Validation Accuracy: {val_results[1]:.4f}")
    else:
        print("Training new baseline model...")
        model_baseline = unet_baseline()
        print(f"Baseline U-Net Parameters: {model_baseline.count_params():,}")
        
        history_baseline = model_baseline.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=4,
            verbose=1
        )
        
        model_baseline.save(baseline_model_path)
        print("✓ Baseline model saved")
else:
    print("No existing model found. Training new baseline model...")
    model_baseline = unet_baseline()
    print(f"Baseline U-Net Parameters: {model_baseline.count_params():,}")
    
    history_baseline = model_baseline.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=4,
        verbose=1
    )
    
    model_baseline.save(baseline_model_path)
    print("✓ Baseline model saved")

print()

# ================================================================
# SECTION 5: TRAIN U-NET WITH ATTENTION
# ================================================================
print("="*60)
print("SECTION 5: U-NET WITH ATTENTION")
print("="*60)

attention_model_path = 'models/unet_attention.h5'

# Check if model already exists
if os.path.exists(attention_model_path):
    print(f"✓ Found existing model: {attention_model_path}")
    response = input("Do you want to use the existing model? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("Loading existing attention model...")
        model_attention = tf.keras.models.load_model(attention_model_path)
        print("✓ Attention model loaded successfully")
        
        # Evaluate to get history-like metrics
        print("Evaluating model on validation set...")
        val_results = model_attention.evaluate(X_val, y_val, verbose=0)
        
        # Create a mock history object for plotting
        history_attention = type('obj', (object,), {
            'history': {
                'accuracy': [val_results[1]] * 10,
                'val_accuracy': [val_results[1]] * 10,
                'loss': [val_results[0]] * 10,
                'val_loss': [val_results[0]] * 10,
                'precision': [val_results[2]] * 10,
                'val_precision': [val_results[2]] * 10,
                'recall': [val_results[3]] * 10,
                'val_recall': [val_results[3]] * 10
            }
        })()
        print(f"Validation Accuracy: {val_results[1]:.4f}")
    else:
        print("Training new attention model...")
        model_attention = unet_attention()
        print(f"Attention U-Net Parameters: {model_attention.count_params():,}")
        
        history_attention = model_attention.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=4,
            verbose=1
        )
        
        model_attention.save(attention_model_path)
        print("✓ Attention model saved")
else:
    print("No existing model found. Training new attention model...")
    model_attention = unet_attention()
    print(f"Attention U-Net Parameters: {model_attention.count_params():,}")
    
    history_attention = model_attention.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=4,
        verbose=1
    )
    
    model_attention.save(attention_model_path)
    print("✓ Attention model saved")

print()

# ================================================================
# SECTION 6: VISUALIZE SEGMENTATION RESULTS
# ================================================================
print("="*60)
print("SECTION 6: VISUALIZING SEGMENTATION RESULTS")
print("="*60)

def plot_segmentation_comparison(images, masks_true, masks_baseline, masks_attention, n=5):
    """Plot comparison of baseline vs attention segmentation"""
    fig, axes = plt.subplots(n, 5, figsize=(20, 4*n))
    
    for i in range(n):
        # Original Image
        axes[i, 0].imshow(images[i].reshape(256, 256), cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground Truth Mask
        axes[i, 1].imshow(masks_true[i].reshape(256, 256), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Baseline Prediction
        axes[i, 2].imshow(masks_baseline[i].reshape(256, 256), cmap='gray')
        axes[i, 2].set_title('Baseline U-Net')
        axes[i, 2].axis('off')
        
        # Attention Prediction
        axes[i, 3].imshow(masks_attention[i].reshape(256, 256), cmap='gray')
        axes[i, 3].set_title('Attention U-Net')
        axes[i, 3].axis('off')
        
        # Overlay Comparison
        axes[i, 4].imshow(images[i].reshape(256, 256), cmap='gray')
        axes[i, 4].imshow(masks_attention[i].reshape(256, 256), cmap='jet', alpha=0.5)
        axes[i, 4].set_title('Attention Overlay')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/segmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: results/segmentation_comparison.png")

# Generate predictions
pred_baseline = model_baseline.predict(X_val[:5])
pred_attention = model_attention.predict(X_val[:5])

# Binarize predictions
pred_baseline = (pred_baseline > 0.5).astype(np.uint8)
pred_attention = (pred_attention > 0.5).astype(np.uint8)

# Plot comparison
plot_segmentation_comparison(X_val[:5], y_val[:5], pred_baseline, pred_attention)
print()

# ================================================================
# SECTION 7: LOAD CLINICAL DATA FOR CLASSIFICATION
# ================================================================
print("="*60)
print("SECTION 7: PREPARING CLASSIFICATION DATA")
print("="*60)

# Load segmented masks for classification
def load_segmented_masks(masks_path, target_size=(256, 256)):
    """Load segmented masks for classification"""
    masks = []
    filenames = []
    
    mask_files = [f for f in os.listdir(masks_path) if f.endswith('_mask.png')]
    
    for mask_file in mask_files:
        mask_path = os.path.join(masks_path, mask_file)
        mask = load_img(mask_path, color_mode='grayscale', target_size=target_size)
        mask = img_to_array(mask) / 255.0
        
        masks.append(mask)
        filenames.append(mask_file.replace('_mask.png', '.txt'))
    
    return np.array(masks), filenames

# Load clinical data CSV
# UPDATE THIS PATH to your actual CSV file location
csv_file_path = "data/output.csv"  # Change this to your CSV path

# If CSV doesn't exist, create it from clinical readings
if not os.path.exists(csv_file_path):
    print("⚠ CSV file not found.")
    print("Options:")
    print("1. Place your CSV at: data/output.csv")
    print("2. Use dummy labels (for testing)")
    
    use_dummy = input("Use dummy labels? (yes/no): ").strip().lower()
    
    if use_dummy in ['yes', 'y']:
        print("Using dummy labels for demonstration...")
        # Create dummy data - 70% normal, 30% abnormal (more realistic)
        np.random.seed(42)
        diseases = np.random.choice(
            ['normal', 'abnormal'], 
            size=len(filenames),
            p=[0.7, 0.3]  # 70% normal, 30% abnormal
        )
        patient_data = pd.DataFrame({
            'File Name': [f.replace('.png', '.txt') for f in filenames],
            'Disease': diseases
        })
        print("✓ Dummy data created")
    else:
        print("Please create CSV file and run again.")
        exit()
else:
    patient_data = pd.read_csv(csv_file_path)
    patient_data['File Name'] = patient_data['File Name'].str.replace('.png', '.txt')

# Load masks and merge with patient data
masks_clf, mask_filenames = load_segmented_masks(MASKS_PATH)
mask_df = pd.DataFrame(mask_filenames, columns=['File Name'])
merged_data = pd.merge(mask_df, patient_data, on='File Name', how='inner')

# Filter masks to only include those with matching labels
matched_indices = []
for idx, filename in enumerate(mask_filenames):
    if filename in merged_data['File Name'].values:
        matched_indices.append(idx)

# Keep only matched masks
masks_clf = masks_clf[matched_indices]

# Encode labels (now aligned with masks)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(merged_data['Disease'])

print(f"Total masks found: {len(mask_filenames)}")
print(f"Matched with labels: {len(masks_clf)}")
print(f"Classification samples: {len(masks_clf)}")
print(f"Label distribution: {dict(zip(*np.unique(labels_encoded, return_counts=True)))}")

# Verify alignment
assert len(masks_clf) == len(labels_encoded), f"Mismatch: {len(masks_clf)} masks vs {len(labels_encoded)} labels"
print("✓ Data alignment verified")
print()

# ================================================================
# SECTION 8: CNN CLASSIFICATION MODEL
# ================================================================
print("="*60)
print("SECTION 8: CNN CLASSIFICATION")
print("="*60)

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    masks_clf, labels_encoded, test_size=0.2, random_state=42
)

# Convert to categorical
y_train_clf = tf.keras.utils.to_categorical(y_train_clf, num_classes=len(label_encoder.classes_))
y_test_clf = tf.keras.utils.to_categorical(y_test_clf, num_classes=len(label_encoder.classes_))

def build_cnn(input_shape, num_classes):
    """Build CNN for classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

cnn_model_path = 'models/cnn_classifier.h5'

# Check if model already exists
if os.path.exists(cnn_model_path):
    print(f"✓ Found existing model: {cnn_model_path}")
    response = input("Do you want to use the existing model? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("Loading existing CNN model...")
        try:
            model_cnn = tf.keras.models.load_model(cnn_model_path)
            
            # Check if model output matches current data
            model_output_shape = model_cnn.output_shape[-1]
            current_num_classes = len(label_encoder.classes_)
            
            if model_output_shape != current_num_classes:
                print(f"⚠ Model mismatch: saved model has {model_output_shape} classes, but current data has {current_num_classes} classes")
                print("Training new model instead...")
                raise ValueError("Class mismatch")
            
            print("✓ CNN model loaded successfully")
            
            # Evaluate to get history-like metrics
            print("Evaluating model on test set...")
            test_results = model_cnn.evaluate(X_test_clf, y_test_clf, verbose=0)
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("Training new model...")
            response = 'no'  # Force retraining
    if response in ['yes', 'y']:
        print("Loading existing CNN model...")
        try:
            model_cnn = tf.keras.models.load_model(cnn_model_path)
            
            # Check if model output matches current data
            model_output_shape = model_cnn.output_shape[-1]
            current_num_classes = len(label_encoder.classes_)
            
            if model_output_shape != current_num_classes:
                print(f"⚠ Model mismatch: saved model has {model_output_shape} classes, but current data has {current_num_classes} classes")
                print("Training new model instead...")
                raise ValueError("Class mismatch")
            
            print("✓ CNN model loaded successfully")
            
            # Evaluate to get history-like metrics
            print("Evaluating model on test set...")
            test_results = model_cnn.evaluate(X_test_clf, y_test_clf, verbose=0)
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("Training new model...")
            response = 'no'  # Force retraining
    
    if response not in ['yes', 'y']:
        print("Training new CNN model...")
        model_cnn = build_cnn((256, 256, 1), len(label_encoder.classes_))
        print(f"CNN Parameters: {model_cnn.count_params():,}")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Classes: {label_encoder.classes_}")
        
        history_cnn = model_cnn.fit(
            X_train_clf, y_train_clf,
            validation_data=(X_test_clf, y_test_clf),
            epochs=10,
            batch_size=32,
            verbose=1
        )
        
        model_cnn.save(cnn_model_path)
        print("✓ CNN model saved")
    else:
        # Create a mock history object for plotting
        history_cnn = type('obj', (object,), {
            'history': {
                'accuracy': [test_results[1]] * 10,
                'val_accuracy': [test_results[1]] * 10,
                'loss': [test_results[0]] * 10,
                'val_loss': [test_results[0]] * 10,
                'precision': [test_results[2]] * 10,
                'val_precision': [test_results[2]] * 10,
                'recall': [test_results[3]] * 10,
                'val_recall': [test_results[3]] * 10
            }
        })()
        print(f"Test Accuracy: {test_results[1]:.4f}")
else:
    print("No existing model found. Training new CNN model...")
    model_cnn = build_cnn((256, 256, 1), len(label_encoder.classes_))
    print(f"CNN Parameters: {model_cnn.count_params():,}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    history_cnn = model_cnn.fit(
        X_train_clf, y_train_clf,
        validation_data=(X_test_clf, y_test_clf),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    model_cnn.save(cnn_model_path)
    print("✓ CNN model saved")

print()

# ================================================================
# SECTION 9: METRICS COMPARISON
# ================================================================
print("="*60)
print("SECTION 9: METRICS COMPARISON")
print("="*60)

def plot_metrics_comparison(history_baseline, history_attention, history_cnn):
    """Plot comprehensive metrics comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    titles = ['Accuracy', 'Loss', 'Precision', 'Recall']
    
    # Segmentation metrics (Baseline vs Attention)
    for idx, (metric, title) in enumerate(zip(metrics[:4], titles)):
        if idx < 2:
            row, col = 0, idx
        else:
            row, col = 1, idx - 2
        
        ax = axes[row, col]
        
        if metric in history_baseline.history:
            ax.plot(history_baseline.history[metric], label='Baseline U-Net (Train)', marker='o')
            ax.plot(history_baseline.history[f'val_{metric}'], label='Baseline U-Net (Val)', marker='o')
        
        if metric in history_attention.history:
            ax.plot(history_attention.history[metric], label='Attention U-Net (Train)', marker='s')
            ax.plot(history_attention.history[f'val_{metric}'], label='Attention U-Net (Val)', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Segmentation {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Classification metrics
    ax = axes[1, 2]
    ax.plot(history_cnn.history['accuracy'], label='CNN Train', marker='d')
    ax.plot(history_cnn.history['val_accuracy'], label='CNN Val', marker='d')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: results/metrics_comparison.png")

plot_metrics_comparison(history_baseline, history_attention, history_cnn)
print()

# ================================================================
# SECTION 10: FINAL RESULTS SUMMARY
# ================================================================
print("="*60)
print("SECTION 10: FINAL RESULTS SUMMARY")
print("="*60)

def print_final_metrics():
    """Print final comparison metrics"""
    print("\n" + "="*60)
    print("SEGMENTATION METRICS (Validation Set)")
    print("="*60)
    
    # Baseline U-Net
    baseline_acc = history_baseline.history['val_accuracy'][-1]
    baseline_prec = history_baseline.history['val_precision'][-1]
    baseline_rec = history_baseline.history['val_recall'][-1]
    baseline_f1 = 2 * (baseline_prec * baseline_rec) / (baseline_prec + baseline_rec)
    
    print("\nBaseline U-Net:")
    print(f"  Accuracy:  {baseline_acc:.4f}")
    print(f"  Precision: {baseline_prec:.4f}")
    print(f"  Recall:    {baseline_rec:.4f}")
    print(f"  F1-Score:  {baseline_f1:.4f}")
    
    # Attention U-Net
    attention_acc = history_attention.history['val_accuracy'][-1]
    attention_prec = history_attention.history['val_precision'][-1]
    attention_rec = history_attention.history['val_recall'][-1]
    attention_f1 = 2 * (attention_prec * attention_rec) / (attention_prec + attention_rec)
    
    print("\nAttention U-Net:")
    print(f"  Accuracy:  {attention_acc:.4f}")
    print(f"  Precision: {attention_prec:.4f}")
    print(f"  Recall:    {attention_rec:.4f}")
    print(f"  F1-Score:  {attention_f1:.4f}")
    
    # Improvement
    print("\nImprovement with Attention Gates:")
    print(f"  Accuracy:  {(attention_acc - baseline_acc):.4f} ({((attention_acc - baseline_acc) / baseline_acc * 100):.2f}%)")
    print(f"  Precision: {(attention_prec - baseline_prec):.4f} ({((attention_prec - baseline_prec) / baseline_prec * 100):.2f}%)")
    print(f"  Recall:    {(attention_rec - baseline_rec):.4f} ({((attention_rec - baseline_rec) / baseline_rec * 100):.2f}%)")
    print(f"  F1-Score:  {(attention_f1 - baseline_f1):.4f} ({((attention_f1 - baseline_f1) / baseline_f1 * 100):.2f}%)")
    
    # Classification
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS (Test Set)")
    print("="*60)
    
    cnn_acc = history_cnn.history['val_accuracy'][-1]
    cnn_prec = history_cnn.history['val_precision'][-1]
    cnn_rec = history_cnn.history['val_recall'][-1]
    cnn_f1 = 2 * (cnn_prec * cnn_rec) / (cnn_prec + cnn_rec)
    
    print("\nCNN Classifier:")
    print(f"  Accuracy:  {cnn_acc:.4f}")
    print(f"  Precision: {cnn_prec:.4f}")
    print(f"  Recall:    {cnn_rec:.4f}")
    print(f"  F1-Score:  {cnn_f1:.4f}")
    
    print("\n" + "="*60)
    print("✓ PROJECT COMPLETE!")
    print("="*60)
    print("\nSaved Models:")
    print("  - models/unet_baseline.h5")
    print("  - models/unet_attention.h5")
    print("  - models/cnn_classifier.h5")
    print("\nSaved Results:")
    print("  - results/segmentation_comparison.png")
    print("  - results/metrics_comparison.png")
    print("="*60)

print_final_metrics()

# ================================================================
# END OF SCRIPT
# ================================================================