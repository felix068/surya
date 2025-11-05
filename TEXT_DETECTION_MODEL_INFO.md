# 📊 Surya Text Detection Model - Informations Complètes

## 🏗️ Architecture du Modèle

**Type** : EfficientViT pour Segmentation Sémantique (modifié)

### Configuration
- **Nombre de classes** : 2 (text + affinity maps)
- **Canaux d'entrée** : 3 (RGB)
- **Résolution d'entrée** : 512x512 pixels
- **Architecture backbone** : EfficientViT-Large
  - Widths: (32, 64, 128, 256, 512)
  - Depths: (1, 1, 1, 6, 6)
  - Hidden sizes: (32, 64, 160, 256)
  - Strides: (2, 2, 2, 2, 2)
  - Head dim: 32

### Décodeur
- **Decoder hidden size** : 512
- **Decoder layer hidden size** : 128

## 📈 Taille du Modèle

### Paramètres
- **Total de paramètres** : **38,424,866** (~38.4M)
- **Paramètres entraînables** : 38,424,866 (100%)

### Taille sur disque
- **FP32** : 146.58 MB
- **FP16** : 73.29 MB

## 🎓 Entraînement

### Hardware
- **GPUs** : 4x NVIDIA A6000 (48GB VRAM chacun)
- **Durée d'entraînement** : **3 jours**

### Méthode
- **Training from scratch** (pas de préentraînement)
- Architecture modifiée d'EfficientViT pour la segmentation sémantique

### Dataset d'entraînement
- **Composition** : "Diverse set of images" (ensemble diversifié d'images)
- **Disponibilité publique** : ❌ **NON PUBLIC** - Le dataset exact n'est pas mentionné dans le README
- **Note** : L'auteur ne partage pas publiquement le dataset d'entraînement, seulement les poids du modèle

## 🧪 Dataset de Benchmark (Test uniquement)

Le modèle est **évalué** (mais pas entraîné) sur :
- **Dataset** : `vikp/doclaynet_bench` sur HuggingFace
- **Source** : Sous-ensemble de [DocLayNet](https://huggingface.co/datasets/vikp/doclaynet_bench)
- **Format** : Images avec bounding boxes annotées à 1000x1000 normalisées

### Autres datasets de benchmark mentionnés
- Layout : `vikp/publaynet_bench` (Publaynet)
- Reading order : `vikp/order_bench`
- Table rec : `datalab-to/fintabnet_bench`
- Recognition : `vikp/rec_bench`

## 🔧 Résolution et Preprocessing

### Résolution d'entrée du modèle
- **Taille fixe** : 512x512 pixels
- **Normalisation** : ImageNet stats
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Gestion des grandes images
- **Seuil de découpe** : Images > 1400px de hauteur
- **Méthode** : Split vertical en chunks de 512px
- **Padding** : Padding blanc (255) pour le dernier chunk si nécessaire

## ⚡ Performance

### Vitesse (GPU A10)
- **Temps par page** : 0.094 secondes
- **Throughput** : ~10.6 pages/seconde
- **Batch size par défaut** : 36 (utilise ~16GB VRAM)

### Vitesse (CPU - 32 cores)
- **Batch size par défaut** : 8

### Métriques de qualité
- **Precision** : 0.836
- **Recall** : 0.961
- **Dataset** : DocLayNet benchmark

### Comparaison avec Tesseract
| Modèle    | Temps/page | Precision | Recall   |
|-----------|------------|-----------|----------|
| Surya     | 0.094s     | 0.836     | 0.961    |
| Tesseract | 0.291s     | 0.631     | 0.998    |

## 🚀 Compilation (Optimisation)

Le modèle supporte la compilation PyTorch :
- **Flag** : `COMPILE_DETECTOR=true`
- **Speedup sur A10** : ~3.3% plus rapide
- **Backend** : OpenXLA pour TPU, sinon backend par défaut

## 🔗 Checkpoint du Modèle

### Localisation
- **URL S3** : `s3://text_detection/2025_05_07`
- **Base URL** : `https://models.datalab.to/text_detection/2025_05_07`
- **Cache local** : `~/.cache/datalab/models/text_detection/2025_05_07`

### Fichiers du modèle
- `config.json` - Configuration du modèle
- `model.safetensors` - Poids du modèle (format SafeTensors)
- `preprocessor_config.json` - Configuration du préprocesseur
- `manifest.json` - Liste des fichiers

## 📝 Licence

- **Modèle** : AI Pubs Open Rail-M (modifiée)
  - Gratuit pour : recherche, usage personnel, startups <$2M
  - Commercial : licence payante requise
- **Code** : GPL v3

## 🎯 Utilisation

### Batch size recommendations
| Device | Batch Size | VRAM Usage |
|--------|-----------|------------|
| CPU    | 8         | N/A        |
| MPS    | 8         | N/A        |
| CUDA   | 36        | ~16 GB     |
| XLA    | 18        | Variable   |

### Chaque batch item utilise
- **~440 MB de VRAM** par image (à 512x512)

## 🔬 Architecture Technique Détaillée

### Encoder (EfficientViT-Large)
1. **Input Stem** : ConvNormAct avec stride 2
2. **Stage 1-3** : Convolutions dépthwise + MBConv
3. **Stage 4** : EfficientVitBlocks avec LiteMLA (attention légère multi-échelle)
4. **Output** : Features à différentes échelles pour le décodeur

### Decoder (DecodeHead)
1. **MLPs** : Unification des dimensions de canaux
2. **Upsampling** : Interpolation bilinéaire
3. **Concatenation** : Fusion des features multi-échelles
4. **Classification Head** : Conv2d → 2 canaux (text + affinity)
5. **Activation** : Sigmoid → [0, 1]

### Post-processing
1. **Seuillage dynamique** basé sur statistiques d'image
2. **Connected components** (OpenCV)
3. **Fit de rectangles tournés** (`cv2.minAreaRect`)
4. **Filtrage** par taille minimum (10 pixels)
5. **Nettoyage** : suppression des boxes contenues dans d'autres
6. **Expansion verticale** : +5% pour capturer les lignes complètes

## 📚 Références

- **Paper EfficientViT** : https://arxiv.org/abs/2205.14756
- **Code original** : https://github.com/mit-han-lab/efficientvit
- **CRAFT** (inspiration) : https://github.com/clovaai/CRAFT-pytorch
- **Repo Surya** : https://github.com/VikParuchuri/surya

---

**Créé le** : 2025-11-05
**Version du modèle** : 2025_05_07
