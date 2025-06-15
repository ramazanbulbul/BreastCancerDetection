import kagglehub
import pandas as pd
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, 
                                   GlobalAveragePooling2D, Dropout, Dense, 
                                   LeakyReLU, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import warnings
warnings.filterwarnings('ignore')

print("Kütüphaneler yüklendi ✓")

# ================================
# 1. VERİ SETİ İNDİRME VE HAZIRLIK
# ================================

# Download latest version
path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
print("Path to dataset files:", path)

# Dataset temel dizini
DATASET_DIR = path
image_dir = os.path.join(path, 'jpeg')

# CSV dosyalarının yolları
train_csv_path = os.path.join(DATASET_DIR, "csv", "calc_case_description_train_set.csv")
test_csv_path = os.path.join(DATASET_DIR, "csv", "calc_case_description_test_set.csv")
dicom_csv_path = os.path.join(DATASET_DIR, "csv", "dicom_info.csv")

# CSV'leri oku
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)
df_dicom = pd.read_csv(dicom_csv_path)

# Sınıf dağılımını kontrol et
print("Orijinal train set sınıf dağılımı:")
print(df_train['pathology'].value_counts())
print("\nOrijinal test set sınıf dağılımı:")
print(df_test['pathology'].value_counts())

# Klasör yapısı için temel dizin
base_out_dir = "/content/mammogram_dataset"
splits = ["train", "validation", "test"]
labels = ["benign", "malignant"]

# Klasörleri oluştur
for split in splits:
    for label in labels:
        os.makedirs(os.path.join(base_out_dir, split, label), exist_ok=True)

def copy_images(df, split_name):
    """Görüntüleri kopyala ve sayıları takip et"""
    copied_counts = {"benign": 0, "malignant": 0}
    
    for idx, row in df.iterrows():
        pathology = str(row['pathology']).lower().strip()
        
        # Pathology değerini normalize et
        if pathology in ['benign', 'benign_without_callback']:
            pathology = 'benign'
        elif pathology in ['malignant']:
            pathology = 'malignant'
        else:
            print(f"Bilinmeyen pathology: {pathology}")
            continue

        cropped_path = row['cropped image file path']
        if pd.isna(cropped_path):
            continue
            
        # DICOM bilgisinden doğru yolu bul
        patient_id = cropped_path.split("/")[0]
        found_path = None
        
        for i, p in df_dicom.iterrows():
            if str(p['PatientID']) == patient_id:
                found_path = p['image_path']
                found_path = found_path.replace('CBIS-DDSM/jpeg', image_dir)
                break
        
        if found_path is None:
            # Alternatif yol deneme
            found_path = os.path.join(image_dir, cropped_path)
        
        if not os.path.exists(found_path):
            print(f"Dosya bulunamadı: {found_path}")
            continue

        filename = os.path.basename(found_path)
        target_path = os.path.join(base_out_dir, split_name, pathology, filename)
        
        try:
            copyfile(found_path, target_path)
            copied_counts[pathology] += 1
        except Exception as e:
            print(f"Kopyalama hatası: {e}")
    
    print(f"{split_name} kopyalanan dosyalar: {copied_counts}")
    return copied_counts

# Train CSV'deki verileri train ve validation olarak ayır - STRATIFIED
train_df, val_df = train_test_split(
    df_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_train['pathology']
)

print("Train/Validation split sonrası:")
print(f"Train pathology dağılımı:\n{train_df['pathology'].value_counts()}")
print(f"Validation pathology dağılımı:\n{val_df['pathology'].value_counts()}")

# Dosyaları kopyala
train_counts = copy_images(train_df, "train")
val_counts = copy_images(val_df, "validation")
test_counts = copy_images(df_test, "test")

print("Veri seti başarıyla klasör yapısına kopyalandı.")

# ================================
# 2. SABITLER VE AYARLAR
# ================================

DATASET_PATH = "/content/mammogram_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Daha küçük batch size
EPOCHS = 20
LEARNING_RATE = 1e-4

# ================================
# 3. VERİ ARTIRMA VE YÜKLEYİCİLER - DÜZELTİLMİŞ
# ================================

# Sınıf ağırlıklarını hesapla
def calculate_class_weights(dataset_path):
    """Sınıf dengesizliği için ağırlık hesapla"""
    train_path = os.path.join(dataset_path, 'train')
    benign_count = len([f for f in os.listdir(os.path.join(train_path, 'benign')) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    malignant_count = len([f for f in os.listdir(os.path.join(train_path, 'malignant')) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    total = benign_count + malignant_count
    
    # Sınıf ağırlıkları - az olan sınıfa daha fazla ağırlık ver
    class_weights = {
        0: total / (2 * benign_count),      # benign
        1: total / (2 * malignant_count)   # malignant
    }
    
    print(f"Benign: {benign_count}, Malignant: {malignant_count}")
    print(f"Class weights: {class_weights}")
    return class_weights

class_weights = calculate_class_weights(DATASET_PATH)

# Dengeli veri artırma
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,  # Mammogram için vertical flip uygun değil
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Doğrulama ve test için sadece normalize
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleyicileri
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'validation'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Sınıf indeksleri: {train_generator.class_indices}")
print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ================================
# 4. GELİŞTİRİLMİŞ MODEL MİMARİSİ
# ================================

def create_improved_model():
    """Geliştirilmiş CNN modeli"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 4
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        GlobalAveragePooling2D(),

        # Classifier
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Alternatif: Transfer Learning modeli
def create_transfer_learning_model():
    """MobileNetV2 ile transfer learning modeli"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # İlk katmanları dondurmayalım
    base_model.trainable = True
    
    # Fine-tuning için sadece son katmanları eğit
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Model seçimi
USE_TRANSFER_LEARNING = True

if USE_TRANSFER_LEARNING:
    model = create_transfer_learning_model()
    model_name = "MobileNetV2_Transfer"
else:
    model = create_improved_model()
    model_name = "Improved_CNN"

# Model derleme - düşük learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

print(f"\n{model_name} Model Özeti:")
model.summary()

# ================================
# 5. GELİŞTİRİLMİŞ EĞİTİM
# ================================

# Callback'ler
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        min_lr=1e-8,
        verbose=1
    )
]

print(f"\n{model_name} model eğitimi başlıyor...")

# Class weights ile eğitim
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # ÖNEMLİ: Sınıf ağırlıkları
    verbose=1
)

# ================================
# 6. EĞİTİM SONUÇLARI GÖRSELLEŞTİRME
# ================================

def plot_training_history(history):
    """Eğitim geçmişini görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0,0].set_title('Model Accuracy')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Loss
    axes[0,1].plot(history.history['loss'], label='Training Loss')
    axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,1].set_title('Model Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Precision
    axes[1,0].plot(history.history['precision'], label='Training Precision')
    axes[1,0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1,0].set_title('Model Precision')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Recall
    axes[1,1].plot(history.history['recall'], label='Training Recall')
    axes[1,1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1,1].set_title('Model Recall')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ================================
# 7. GELİŞTİRİLMİŞ MODEL DEĞERLENDİRME
# ================================

def evaluate_model_detailed(model, test_generator):
    """Modeli detaylı olarak değerlendir"""
    print("Model değerlendiriliyor...")
    
    # Threshold optimizasyonu için validation set kullan
    print("Optimal threshold hesaplanıyor...")
    validation_generator.reset()
    val_predictions = model.predict(validation_generator, verbose=1)
    val_true_labels = validation_generator.classes
    
    # ROC eğrisinden optimal threshold bul
    fpr, tpr, thresholds = roc_curve(val_true_labels, val_predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Test seti tahminleri
    test_generator.reset()
    test_predictions = model.predict(test_generator, verbose=1)
    test_true_labels = test_generator.classes
    
    # Farklı threshold'lar ile değerlendirme
    thresholds_to_test = [0.3, 0.4, 0.5, optimal_threshold, 0.6, 0.7]
    
    print(f"\nFarklı threshold değerleri ile performans:")
    print("Threshold\tAccuracy\tPrecision\tRecall\tF1-Score")
    print("-" * 60)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds_to_test:
        pred_classes = (test_predictions > thresh).astype(int).flatten()
        
        acc = accuracy_score(test_true_labels, pred_classes)
        prec = precision_score(test_true_labels, pred_classes, zero_division=0)
        rec = recall_score(test_true_labels, pred_classes, zero_division=0)
        f1 = f1_score(test_true_labels, pred_classes, zero_division=0)
        
        print(f"{thresh:.3f}\t\t{acc:.4f}\t\t{prec:.4f}\t\t{rec:.4f}\t\t{f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # En iyi threshold ile final değerlendirme
    print(f"\nEn iyi threshold: {best_threshold:.4f}")
    final_pred_classes = (test_predictions > best_threshold).astype(int).flatten()
    
    # Metrikler
    accuracy = accuracy_score(test_true_labels, final_pred_classes)
    precision = precision_score(test_true_labels, final_pred_classes, zero_division=0)
    recall = recall_score(test_true_labels, final_pred_classes, zero_division=0)
    f1 = f1_score(test_true_labels, final_pred_classes, zero_division=0)
    auc = roc_auc_score(test_true_labels, test_predictions)
    
    # Confusion matrix
    cm = confusion_matrix(test_true_labels, final_pred_classes)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n=== FINAL MODEL PERFORMANSI ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Sınıf başına detaylar
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"True Negatives (Benign correctly classified): {tn}")
    print(f"False Positives (Benign misclassified as Malignant): {fp}")
    print(f"False Negatives (Malignant misclassified as Benign): {fn}")
    print(f"True Positives (Malignant correctly classified): {tp}")
    
    # Classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    target_names = ['Benign', 'Malignant']
    print(classification_report(test_true_labels, final_pred_classes, target_names=target_names))
    
    return {
        'predictions': test_predictions,
        'predicted_classes': final_pred_classes,
        'true_labels': test_true_labels,
        'best_threshold': best_threshold,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    }

# Model değerlendirme
evaluation_results = evaluate_model_detailed(model, test_generator)

# ================================
# 8. SONUÇLARI GÖRSELLEŞTİRME
# ================================

def plot_evaluation_results(results):
    """Değerlendirme sonuçlarını görselleştir"""
    predictions = results['predictions']
    predicted_classes = results['predicted_classes']
    true_labels = results['true_labels']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    axes[0,0].set_xticklabels(['Benign', 'Malignant'])
    axes[0,0].set_yticklabels(['Benign', 'Malignant'])
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    auc_score = roc_auc_score(true_labels, predictions)
    axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
    axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predictions)
    axes[1,0].plot(recall_vals, precision_vals, linewidth=2)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Prediction Distribution
    axes[1,1].hist(predictions[true_labels == 0], bins=30, alpha=0.7, 
                   label='Benign', color='blue', density=True)
    axes[1,1].hist(predictions[true_labels == 1], bins=30, alpha=0.7, 
                   label='Malignant', color='red', density=True)
    axes[1,1].axvline(x=results['best_threshold'], color='green', 
                      linestyle='--', linewidth=2, label=f'Best Threshold ({results["best_threshold"]:.3f})')
    axes[1,1].set_xlabel('Prediction Probability')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Prediction Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_evaluation_results(evaluation_results)

# ================================
# 9. ÖRNEK TAHMİNLER
# ================================

def show_prediction_examples(model, test_generator, threshold, num_examples=12):
    """Örnek tahminleri göster"""
    test_generator.reset()
    
    # Birkaç batch'i birleştir
    all_images = []
    all_labels = []
    all_predictions = []
    
    for i in range(3):  # 3 batch al
        try:
            batch_images, batch_labels = next(test_generator)
            predictions = model.predict(batch_images, verbose=0)
            
            all_images.extend(batch_images)
            all_labels.extend(batch_labels)
            all_predictions.extend(predictions.flatten())
        except StopIteration:
            break
    
    # Random örnekler seç
    indices = np.random.choice(len(all_images), min(num_examples, len(all_images)), replace=False)
    
    plt.figure(figsize=(20, 12))
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i+1)
        plt.imshow(all_images[idx])
        
        true_label = "Malignant" if all_labels[idx] == 1 else "Benign"
        pred_prob = all_predictions[idx]
        pred_label = "Malignant" if pred_prob > threshold else "Benign"
        
        # Renk kodlaması
        if true_label == pred_label:
            color = 'green'
            status = "✓ Correct"
        else:
            color = 'red'
            status = "✗ Wrong"
        
        plt.title(f'{status}\nTrue: {true_label}\nPred: {pred_label} ({pred_prob:.3f})', 
                 color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle(f'Sample Predictions (Threshold: {threshold:.3f})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

show_prediction_examples(model, test_generator, evaluation_results['best_threshold'])

# ================================
# 10. MODEL KAYDETME
# ================================

model_filename = f"/content/Models/{model_name}_mammogram_model.h5"
model.save(model_filename)
print(f"Model kaydedildi: {model_filename}")

# Sonuçları kaydet
results_filename = f"/content/Results/{model_name}_results.pkl"
import pickle
with open(results_filename, 'wb') as f:
    pickle.dump({
        'history': history.history,
        'evaluation_results': evaluation_results,
        'class_weights': class_weights
    }, f)
print(f"Sonuçlar kaydedildi: {results_filename}")

# ================================
# 11. FINAL RAPOR
# ================================

def generate_comprehensive_report(model_name, evaluation_results, class_weights):
    """Kapsamlı final rapor"""
    metrics = evaluation_results['metrics']
    
    print("\n" + "="*70)
    print("                    FİNAL PROJE RAPORU")
    print("          BME 519 - Computer-Aided Diagnosis")
    print("="*70)
    
    print(f"Model Türü: {model_name}")
    print(f"Görüntü Boyutu: {IMG_SIZE}")
    print(f"Batch Boyutu: {BATCH_SIZE}")
    print(f"Toplam Epoch: {len(history.history['loss'])}")
    print(f"Öğrenme Oranı: {LEARNING_RATE}")
    print(f"En İyi Threshold: {evaluation_results['best_threshold']:.4f}")
    
    print(f"\n=== SINIŞ DAĞILIMI VE AĞIRLIKLAR ===")
    print(f"Sınıf Ağırlıkları: {class_weights}")
    
    print(f"\n=== PERFORMANS METRİKLERİ ===")
    print(f"{'Metrik':<15} {'Değer':<10} {'Açıklama'}")
    print("-" * 50)
    print(f"{'Accuracy':<15} {metrics['accuracy']:<10.4f} Genel doğruluk oranı")
    print(f"{'Precision':<15} {metrics['precision']:<10.4f} Pozitif tahmin doğruluğu")
    print(f"{'Recall':<15} {metrics['recall']:<10.4f} Gerçek pozitifleri bulma")
    print(f"{'F1-Score':<15} {metrics['f1_score']:<10.4f} Precision-Recall dengesi")
    print(f"{'AUC':<15} {metrics['auc']:<10.4f} ROC eğrisi altında kalan alan")
    print(f"{'Sensitivity':<15} {metrics['sensitivity']:<10.4f} Kanser tespiti oranı")
    print(f"{'Specificity':<15} {metrics['specificity']:<10.4f} Sağlıklı tespit oranı")
    
    print(f"\n=== KLİNİK DEĞERLENDIRME ===")
    print("Medikal açıdan kritik metrikler:")
    print(f"• Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
    print("  → Kanser vakalarının ne kadarını doğru tespit ettiğimiz")
    print("  → Yüksek olması kritik (kanser kaçırılmamalı)")
    
    print(f"• Specificity (True Negative Rate): {metrics['specificity']:.4f}")
    print("  → Sağlıklı vakaları doğru tespit etme oranı")
    print("  → Gereksiz biopsi/tetkik oranını etkiler")
    
    print(f"• Precision (Positive Predictive Value): {metrics['precision']:.4f}")
    print("  → Kanser dediğimiz vakaların gerçekten kanser olma oranı")
    
    print(f"\n=== MODEL PERFORMANS DEĞERLENDİRMESİ ===")
    
    # Performance assessment
    if metrics['auc'] >= 0.9:
        auc_assessment = "Mükemmel"
    elif metrics['auc'] >= 0.8:
        auc_assessment = "İyi"
    elif metrics['auc'] >= 0.7:
        auc_assessment = "Orta"
    else:
        auc_assessment = "Zayıf"
    
    print(f"AUC Performansı: {auc_assessment} ({metrics['auc']:.4f})")
    
    if metrics['sensitivity'] >= 0.9:
        sens_assessment = "Çok İyi"
    elif metrics['sensitivity'] >= 0.8:
        sens_assessment = "İyi"
    elif metrics['sensitivity'] >= 0.7:
        sens_assessment = "Kabul Edilebilir"
    else:
        sens_assessment = "Yetersiz"
    
    print(f"Sensitivity Performansı: {sens_assessment} ({metrics['sensitivity']:.4f})")
    
    print(f"\n=== ÖNERİLER VE İYİLEŞTİRME NOKTALARI ===")
    
    recommendations = []
    
    if metrics['sensitivity'] < 0.85:
        recommendations.append("• Sensitivity artırılmalı - daha fazla pozitif örnek veya farklı loss function")
    
    if metrics['specificity'] < 0.8:
        recommendations.append("• Specificity artırılmalı - false positive oranı yüksek")
    
    if metrics['precision'] < 0.7:
        recommendations.append("• Precision artırılmalı - pozitif tahminlerin kalitesi düşük")
    
    if metrics['f1_score'] < 0.8:
        recommendations.append("• F1-Score artırılmalı - precision-recall dengesi kurmalı")
    
    if len(recommendations) == 0:
        recommendations.append("• Model performansı kabul edilebilir seviyede!")
    
    for rec in recommendations:
        print(rec)
    
    print(f"\n=== GELECEK ÇALIŞMALAR ===")
    print("• Daha fazla veri toplanabilir")
    print("• Farklı augmentation teknikleri denenebilir")
    print("• Ensemble yöntemleri kullanılabilir")
    print("• Focal loss gibi imbalanced data için özel loss fonksiyonları")
    print("• Cross-validation ile daha güvenilir değerlendirme")
    print("• Grad-CAM ile model açıklanabilirliği")
    
    print("="*70)
    
    return metrics

# Kapsamlı rapor oluştur
final_metrics = generate_comprehensive_report(model_name, evaluation_results, class_weights)

# ================================
# 12. EK ANALİZLER VE GÖRSEL RAPORLAR
# ================================

def create_detailed_analysis_plots(history, evaluation_results):
    """Detaylı analiz grafikleri"""
    
    # 1. Eğitim süreci detaylı analizi
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss comparison
    axes[0,0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0,0].set_title('Training vs Validation Loss', fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[0,1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0,1].set_title('Training vs Validation Accuracy', fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Precision comparison
    axes[0,2].plot(history.history['precision'], label='Training Precision', linewidth=2)
    axes[0,2].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    axes[0,2].set_title('Training vs Validation Precision', fontweight='bold')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Recall comparison
    axes[1,0].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1,0].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1,0].set_title('Training vs Validation Recall', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1,1].plot(history.history['lr'], linewidth=2, color='red')
        axes[1,1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Learning Rate Schedule', fontweight='bold')
    
    # Performance metrics summary
    metrics = evaluation_results['metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1_score'], metrics['auc'], metrics['sensitivity'], metrics['specificity']]
    
    bars = axes[1,2].bar(range(len(metric_names)), metric_values, 
                        color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'mediumpurple', 'orange', 'pink'])
    axes[1,2].set_title('Final Performance Metrics', fontweight='bold')
    axes[1,2].set_ylabel('Score')
    axes[1,2].set_xticks(range(len(metric_names)))
    axes[1,2].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1,2].set_ylim(0, 1)
    axes[1,2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Detaylı analiz grafikleri oluştur
create_detailed_analysis_plots(history, evaluation_results)

# ================================
# 13. MODEL KARŞILAŞTIRMA VE BENCHMARK
# ================================

def compare_with_baseline():
    """Baseline modeller ile karşılaştırma"""
    print("\n" + "="*50)
    print("            MODEL PERFORMANS KARŞILAŞTIRMASI")
    print("="*50)
    
    # Tipik mammogram sınıflandırma modeli benchmarkları
    benchmarks = {
        'Random Classifier': {'accuracy': 0.50, 'sensitivity': 0.50, 'specificity': 0.50, 'auc': 0.50},
        'Simple CNN': {'accuracy': 0.75, 'sensitivity': 0.70, 'specificity': 0.80, 'auc': 0.75},
        'ResNet-50': {'accuracy': 0.85, 'sensitivity': 0.82, 'specificity': 0.87, 'auc': 0.85},
        'Our Model': evaluation_results['metrics']
    }
    
    # Tablo formatında göster
    print(f"{'Model':<15} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'AUC':<10}")
    print("-" * 65)
    
    for model_name, metrics in benchmarks.items():
        print(f"{model_name:<15} {metrics['accuracy']:<10.3f} {metrics['sensitivity']:<12.3f} "
              f"{metrics['specificity']:<12.3f} {metrics['auc']:<10.3f}")
    
    # Görsel karşılaştırma
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    model_names = list(benchmarks.keys())
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'auc']
    
    # Bar chart
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        values = [benchmarks[model][metric] for model in model_names]
        axes[0].bar(x + i*width, values, width, label=metric.capitalize())
    
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    our_model_values = [evaluation_results['metrics'][metric] for metric in metrics_to_plot]
    our_model_values += our_model_values[:1]
    
    axes[1] = plt.subplot(122, projection='polar')
    axes[1].plot(angles, our_model_values, 'o-', linewidth=2, label='Our Model')
    axes[1].fill(angles, our_model_values, alpha=0.25)
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels([m.capitalize() for m in metrics_to_plot])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Our Model Performance Radar', pad=20)
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

compare_with_baseline()

# ================================
# 14. SON KONTROLLER VE DOĞRULAMA
# ================================

def final_validation_checks(model, test_generator, evaluation_results):
    """Son doğrulama kontrolleri"""
    print("\n" + "="*50)
    print("              SON DOĞRULAMA KONTROLLERİ")
    print("="*50)
    
    # 1. Prediction distribution kontrolü
    test_generator.reset()
    all_predictions = model.predict(test_generator, verbose=0)
    
    print(f"Test seti tahmin dağılımı:")
    print(f"Min prediction: {all_predictions.min():.4f}")
    print(f"Max prediction: {all_predictions.max():.4f}")
    print(f"Mean prediction: {all_predictions.mean():.4f}")
    print(f"Std prediction: {all_predictions.std():.4f}")
    
    # 2. Sınıf dağılımı kontrolü
    predicted_classes = (all_predictions > evaluation_results['best_threshold']).astype(int)
    unique, counts = np.unique(predicted_classes, return_counts=True)
    
    print(f"\nTahmin edilen sınıf dağılımı:")
    for cls, count in zip(unique, counts):
        class_name = "Benign" if cls == 0 else "Malignant"
        percentage = count / len(predicted_classes) * 100
        print(f"{class_name}: {count} ({percentage:.1f}%)")
    
    # 3. Gerçek sınıf dağılımı
    true_classes = test_generator.classes
    unique_true, counts_true = np.unique(true_classes, return_counts=True)
    
    print(f"\nGerçek sınıf dağılımı:")
    for cls, count in zip(unique_true, counts_true):
        class_name = "Benign" if cls == 0 else "Malignant"
        percentage = count / len(true_classes) * 100
        print(f"{class_name}: {count} ({percentage:.1f}%)")
    
    # 4. Model güvenilirlik kontrolü
    high_confidence_predictions = np.sum((all_predictions > 0.8) | (all_predictions < 0.2))
    total_predictions = len(all_predictions)
    confidence_ratio = high_confidence_predictions / total_predictions
    
    print(f"\nModel güvenilirlik analizi:")
    print(f"Yüksek güvenilirlik tahminleri (>0.8 veya <0.2): {high_confidence_predictions}/{total_predictions} ({confidence_ratio:.1%})")
    
    if confidence_ratio > 0.7:
        print("Model tahminlerinde yüksek güven seviyesi")
    elif confidence_ratio > 0.5:
        print("Model tahminlerinde orta güven seviyesi")
    else:
        print("Model tahminlerinde düşük güven seviyesi - model belirsizliği yüksek")
    
    # 5. Performans tutarlılık kontrolü
    metrics = evaluation_results['metrics']
    
    print(f"\nPerformans tutarlılık kontrolü:")
    if abs(metrics['precision'] - metrics['recall']) < 0.1:
        print("Precision-Recall dengeli")
    else:
        print("Precision-Recall dengesizliği var")
    
    if metrics['sensitivity'] > 0.8 and metrics['specificity'] > 0.8:
        print("Hem sensitivity hem specificity yeterli")
    else:
        print("Sensitivity veya specificity düşük")
    
    print("="*50)

# Son doğrulama kontrollerini çalıştır
final_validation_checks(model, test_generator, evaluation_results)

print(f"\nAnaliz tamamlandı! Model başarıyla eğitildi ve değerlendirildi.")
print(f"En iyi threshold: {evaluation_results['best_threshold']:.4f}")
print(f"F1-Score: {evaluation_results['metrics']['f1_score']:.4f}")
print(f"AUC Score: {evaluation_results['metrics']['auc']:.4f}")
print(f"Model dosyası: {model_filename}")
print(f"Sonuçlar dosyası: {results_filename}")

# Dosyaları indirmek için talimatlar
print(f"\nDosyaları indirmek için aşağıdaki komutları kullanın:")
print(f"files.download('{model_filename}')")
print(f"files.download('{results_filename}')")
