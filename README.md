# Breast Cancer Detection - MobileNetV2 Transfer Learning Kullanarak Meme Kanseri Tespiti

MobileNetV2 transfer learning tekniği kullanarak mamografi görüntülerinden malign ve benign meme kanseri tespiti için Computer-Aided Diagnosis (CAD) sistemi.

## 📋 Proje Genel Bakış

Bu proje, CBIS-DDSM (Curated Breast Imaging Subset of Digital Database for Screening Mammography) dataset'ini kullanarak meme kanseri tespiti için binary classification sistemi geliştirmektedir. Sistem, mamografi görüntülerini benign veya malignant olarak sınıflandırmak için MobileNetV2 pre-trained model ile transfer learning tekniğini kullanmaktadır.

**Ders**: BME 519 - Computer-Aided Diagnosis Methods forBiomedical Applications  
**Kurum**: İzmir Katip Çelebi Üniversitesi - Fen Bilimleri Enstitüsü  
**Bölüm**: Yazılım Mühendisliği  
**Dönem**: Bahar 2025

## 🎯 Amaçlar

- Otomatik meme kanseri tespit sistemi geliştirmek
- Medikal görüntü sınıflandırmasında transfer learning tekniklerini uygulamak
- Model performansını klinik metrikler kullanarak değerlendirmek
- Medikal tanı desteği için yorumlanabilir sonuçlar sağlamak

## 📊 Dataset Bilgileri

**Dataset**: CBIS-DDSM Breast Cancer Image Dataset
- **Kaynak**: Kaggle/Public Medical Imaging Repository
- **Toplam Görüntü**: 1,546 training + 326 test görüntüsü
- **Görüntü Formatı**: DICOM'dan JPEG'e dönüştürülmüş
- **Görüntü Boyutu**: 224×224 pixel'e yeniden boyutlandırılmış
- **Sınıflar**: 
  - Benign (benign without callback dahil)
  - Malignant

### Sınıf Dağılımı
| Bölüm | Benign | Malignant | Toplam |
|-------|--------|-----------|--------|
| Training | 801 | 435 | 1,236 |
| Validation | 200 | 109 | 309 |
| Test | 197 | 129 | 326 |

## 🏗️ Model Mimarisi

**Base Model**: MobileNetV2 (ImageNet üzerinde pre-trained)
- **Input Shape**: (224, 224, 3)
- **Transfer Learning**: Donmuş base layer'lar ile feature extraction
- **Custom Head**:
  - Global Average Pooling
  - Dropout (0.5)
  - Dense Layer (256 unit)
  - Batch Normalization
  - LeakyReLU Activation
  - Dropout (0.5)
  - Output Dense Layer (1 unit, sigmoid)

**Toplam Parametre**: 2,587,201
- Trainable: 1,534,785
- Non-trainable: 1,052,416

## 🔧 Gereksinimler

```bash
pip install tensorflow>=2.10.0
pip install keras
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install opencv-python
pip install pillow
```

## 📁 Proje Yapısı

```
BreastCancerDetection/
├── main.py                # Ana training ve evaluation script'i
├── README.md              # Proje dokümantasyonu
├── models/                # Kaydedilmiş modeller
│   └── MobileNetV2_Transfer_mammogram_model.h5
├── results/               # Sonuçlar ve grafikler
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── MobileNetV2_Transfer_results.pkl
└── mammogram_dataset/     # Dataset dizini
    ├── train/
    │   ├── benign/
    │   └── malignant/
    ├── validation/
    │   ├── benign/
    │   └── malignant/
    └── test/
        ├── benign/
        └── malignant/
```

## 🚀 Kullanım

### 1. Dataset Hazırlığı
```python
# Script otomatik olarak CBIS-DDSM dataset'ini organize eder
# Dataset'inizi şuraya yerleştirin: /kaggle/input/cbis-ddsm-breast-cancer-image-dataset/
# Veya script'teki path'i değiştirin
```

### 2. Model Training
```bash
python main.py
```

### 3. Model Değerlendirmesi
Script otomatik olarak modeli değerlendirir ve şunları oluşturur:
- Performans metrikleri
- Confusion matrix
- ROC curve
- Training history grafikleri

## 📈 Sonuçlar

### Performans Metrikleri
| Metrik | Değer | Klinik Önemi |
|--------|-------|--------------|
| **Accuracy** | %59.45 | Genel sınıflandırma doğruluğu |
| **Sensitivity (Recall)** | %60.68 | Kanser tespit oranı (kritik) |
| **Specificity** | %58.62 | Sağlıklı doku tanımlama |
| **Precision** | %49.65 | Pozitif tahmin doğruluğu |
| **F1-Score** | %54.62 | Precision-Recall dengesi |
| **AUC** | %65.04 | Model ayırt etme kabiliyeti |

### Confusion Matrix
```
                 Tahmin Edilen
Gerçek    Benign  Malignant
Benign      102       72
Malignant    46       71
```

### Optimal Classification Threshold
- **Threshold**: 0.3000
- **Gerekçe**: Sensitivity'nin kritik olduğu medikal tanı için optimize edilmiş

## 🔍 Klinik Değerlendirme

### Güçlü Yönler
- ✅ Kanser tespiti için orta seviye sensitivity (%60.68)
- ✅ Precision ve recall arasında dengeli yaklaşım
- ✅ Deep learning ile otomatik feature extraction

### Sınırlılıklar
- ⚠️ Genel accuracy iyileştirmeye ihtiyaç duyuyor (%59.45)
- ⚠️ Yüksek false positive oranı (%41.38)
- ⚠️ Klinik kullanım için precision daha yüksek olmalı
- ⚠️ Sınırlı dataset boyutu generalization'ı etkileyebilir

### Klinik Etki
- **True Positives**: 71 kanser vakası doğru şekilde tespit edildi
- **False Negatives**: 46 kanser vakası kaçırıldı (klinik risk)
- **False Positives**: 72 gereksiz biopsi/takip

## 🔬 Teknik Detaylar

### Training Konfigürasyonu
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 16
- **Epochs**: 12 (Early stopping)
- **Class Weights**: Dengesizliği ele almak için uygulandı
  - Benign: 0.872
  - Malignant: 1.172

### Data Augmentation
- Rotation Range: 20°
- Width/Height Shift: 0.2
- Shear Range: 0.2
- Zoom Range: 0.2
- Horizontal Flip: True

## 📊 Model Karşılaştırması

| Yöntem | Doğruluk | Duyarlılık | Özgüllük | AUC |
|--------|----------|------------|----------|-----|
| Random Classifier | 0,500 | 0,500 | 0,500 | 0,500 |
| Basit CNN | 0,750 | 0,700 | 0,800 | 0,750 |
| ResNet-50 | 0,850 | 0,820 | 0,870 | 0,850 |
| Bizim Modelimiz | **0,595** | **0,607** | **0,586** | **0,650** |

## 🚀 Gelecek İyileştirmeler

### Acil İyileştirmeler
1. **Data Augmentation**: Gelişmiş teknikler (Mixup, CutMix)
2. **Model Architecture**: ResNet, EfficientNet veya Vision Transformers denenmeli
3. **Ensemble Methods**: Birden fazla modeli birleştirme
4. **Loss Functions**: Dengesiz data için Focal Loss

### Gelişmiş Yaklaşımlar
1. **Multi-scale Analysis**: Farklı çözünürlüklerde görüntü analizi
2. **Attention Mechanisms**: İlgili görüntü bölgelerine odaklanma
3. **Cross-validation**: Daha güvenilir değerlendirme
4. **External Validation**: Farklı dataset'lerde test



## 📄 Lisans

Bu proje BME 519 ders kapsamında eğitim amaçlı geliştirilmiştir.

## 📞 İletişim

**Yazar**: Ramazan BÜLBÜL  
**Email**: y240237016@ogr.ikcu.edu.tr  
**Ders**: BME 519 - Bilgisayar Destekli Tanı  
**Kurum**: İzmir Katip Çelebi Üniversitesi - Fen Bilimleri Enstitüsü  
**Bölüm**: Yazılım Mühendisliği  
**Öğretim Üyesi**: Dr. Özlem Karabiber Cura

---

## 🔍 Önemli Dosya Açıklamaları

- **`main.py`**: Data preprocessing, model training ve evaluation için tam pipeline
- **`MobileNetV2_Transfer_mammogram_model.h5`**: Training edilmiş model weights
- **`MobileNetV2_Transfer_results.pkl`**: Serialize edilmiş sonuçlar ve metrikler
  
## ⚡ Hızlı Başlangıç

1. Repository'yi clone edin
2. Dependencies yükleyin
3. CBIS-DDSM dataset'ini indirin
4. Çalıştırın: `python main.py`
5. Sonuçları `results/` dizininde kontrol edin

## 🎓 Akademik Bağlam

Bu proje şunların pratik uygulamasını göstermektedir:
- Medikal Görüntülemede Transfer Learning
- Binary Classification için Deep Learning
- Medikal AI Değerlendirme Metrikleri
- Klinik Karar Destek Sistemleri

**Not**: Bu sistem yalnızca araştırma ve eğitim amaçlıdır ve uygun klinik doğrulama ve düzenleyici onay olmadan gerçek medikal tanı için kullanılmamalıdır.
