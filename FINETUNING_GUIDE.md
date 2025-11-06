# 🎯 Guide de Fine-tuning Surya Text Detection

Guide complet pour fine-tuner le modèle Surya sur vos données personnelles.

## 📁 Structure du Dataset

Votre dataset doit suivre cette structure :

```
data/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   ├── img003.png
│   └── ...
└── labels/
    ├── img001.txt
    ├── img002.txt
    ├── img003.txt
    └── ...
```

## 📝 Format des Labels (YOLO)

Chaque fichier `.txt` dans `labels/` correspond à une image et contient une ligne par bbox :

```
classe x_center y_center width height
```

**Toutes les valeurs sont normalisées entre 0 et 1** (divisées par la largeur/hauteur de l'image)

### Exemple

Pour une image de 1000x800 pixels avec 2 lignes de texte :

```
# img001.txt
0 0.5 0.3 0.8 0.05
0 0.5 0.5 0.8 0.05
```

Explication :
- `0` : Classe (toujours 0 pour "text line")
- `0.5` : Centre X (à 50% de la largeur = 500px)
- `0.3` : Centre Y (à 30% de la hauteur = 240px)
- `0.8` : Largeur (80% de la largeur de l'image = 800px)
- `0.05` : Hauteur (5% de la hauteur = 40px)

## 🛠️ Convertir vos annotations au format YOLO

Si vous avez des bboxes en coordonnées absolues `[x_min, y_min, x_max, y_max]` :

```python
def convert_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """Convertit bbox absolue en format YOLO normalisé."""
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return f"0 {x_center} {y_center} {width} {height}"

# Exemple
img_width, img_height = 1000, 800
x_min, y_min, x_max, y_max = 100, 220, 900, 260

yolo_line = convert_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height)
print(yolo_line)  # 0 0.5 0.3 0.8 0.05
```

Si vous avez un JSON avec des annotations :

```python
import json
import os
from PIL import Image

def convert_json_to_yolo(json_path, images_dir, output_labels_dir):
    """
    Convertit un JSON d'annotations en fichiers YOLO.

    Format JSON attendu:
    {
        "img001.jpg": {
            "bboxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        },
        ...
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_labels_dir, exist_ok=True)

    for img_name, annotations in data.items():
        # Obtenir la taille de l'image
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Convertir chaque bbox
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_name)

        with open(label_path, 'w') as f:
            for bbox in annotations['bboxes']:
                x1, y1, x2, y2 = bbox

                # Convertir en YOLO
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                f.write(f"0 {x_center} {y_center} {width} {height}\n")

# Utilisation
convert_json_to_yolo(
    json_path="annotations.json",
    images_dir="data/images",
    output_labels_dir="data/labels"
)
```

## ⚙️ Configuration

Éditez les paramètres dans `simple_fine.py` (classe `Config`) :

```python
class Config:
    # Chemins du dataset
    IMAGES_DIR = "data/images"        # Dossier avec vos images
    LABELS_DIR = "data/labels"        # Dossier avec vos labels YOLO

    # Sortie
    OUTPUT_DIR = "./finetuned_model"  # Où sauvegarder le modèle

    # Hyperparamètres d'entraînement
    EPOCHS = 20                       # Nombre d'epochs (20-50 recommandé)
    BATCH_SIZE = 4                    # Taille du batch (ajuster selon GPU)
    LEARNING_RATE = 1e-5              # Learning rate (1e-5 à 1e-4)
    WEIGHT_DECAY = 1e-4               # Régularisation

    # Paramètres système
    NUM_WORKERS = 4                   # Workers pour DataLoader
    SAVE_EVERY = 5                    # Sauvegarder tous les N epochs
```

### Ajuster selon votre GPU

| GPU | VRAM | Batch Size recommandé |
|-----|------|----------------------|
| RTX 3060 | 12 GB | 2-4 |
| RTX 3090 | 24 GB | 8-12 |
| A100 | 40 GB | 16-24 |
| CPU | - | 1-2 |

## 🚀 Lancer l'entraînement

```bash
# S'assurer que le dataset est prêt
ls data/images/  # Doit contenir vos images
ls data/labels/  # Doit contenir vos fichiers .txt

# Lancer le fine-tuning
python simple_fine.py
```

## 📊 Résultats attendus

```
Loading pretrained Surya model...
Model loaded: 38,424,866 parameters

Loading dataset...
Found 10000 images in data/images
Dataset size: 10000 images
Batches per epoch: 2500

Starting training...
============================================================
Epoch 1/20: 100%|████████| 2500/2500 [12:34<00:00, 3.31it/s, loss=0.234, avg_loss=0.245]

Epoch 1/20
  Average Loss: 0.2451
  Learning Rate: 1.00e-05
  Saved checkpoint to ./finetuned_model/checkpoint_epoch_5

...

Training complete!
Best loss: 0.1123
Final model saved to: ./finetuned_model/final_model
Best model saved to: ./finetuned_model/best_model
```

## 🔍 Tester le modèle fine-tuné

Modifiez `simple_inf.py` pour utiliser votre modèle :

```python
# Dans simple_inf.py, ligne ~16
# AVANT :
MODEL_CHECKPOINT = "s3://text_detection/2025_05_07"

# APRÈS :
MODEL_CHECKPOINT = "./finetuned_model/best_model"
```

Puis testez :

```bash
python simple_inf.py
```

## 📈 Monitoring et Debugging

### Vérifier les masks générés

Ajoutez ce code dans `simple_fine.py` après la création du dataset :

```python
# Après : dataloader = DataLoader(...)

# Visualiser le premier batch
images, text_masks, affinity_masks, names = next(iter(dataloader))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    # Image
    img = images[i].numpy().transpose(1, 2, 0)
    img = img * Config.STD + Config.MEAN
    img = np.clip(img, 0, 1)
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Image {i+1}")

    # Mask
    axes[1, i].imshow(text_masks[i], cmap='gray')
    axes[1, i].set_title(f"Mask {i+1}")

plt.savefig("debug_batch.png")
print("Saved debug visualization to debug_batch.png")
```

### Réduire la loss

Si la loss ne descend pas :

1. **Vérifier les labels** : Les bboxes sont-elles correctes ?
2. **Augmenter le learning rate** : Essayer `1e-4` au lieu de `1e-5`
3. **Plus d'epochs** : Essayer 50-100 epochs
4. **Plus de données** : 10k images c'est bien, mais plus c'est mieux

Si la loss descend trop vite (overfitting) :

1. **Augmenter le weight decay** : Essayer `1e-3`
2. **Data augmentation** : Ajouter des transformations aléatoires
3. **Réduire le learning rate** : Essayer `5e-6`

## 🎨 Data Augmentation (optionnel)

Pour améliorer la généralisation, ajoutez dans `YOLOTextDataset.__getitem__` :

```python
import albumentations as A

# Définir les transformations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
], is_check_shapes=False)

# Dans __getitem__, après avoir créé le mask :
# Appliquer les transformations
augmented = transform(image=np.array(image), mask=text_mask)
image = Image.fromarray(augmented['image'])
text_mask = augmented['mask']
```

## 🐛 Résolution de problèmes

### Erreur : "CUDA out of memory"

Réduire le batch size :
```python
BATCH_SIZE = 2  # ou 1
```

### Erreur : "No such file or directory: data/images"

Créer les dossiers :
```bash
mkdir -p data/images data/labels
```

### Loss = NaN

Vérifier que :
1. Les labels sont bien entre 0 et 1
2. Les images ne contiennent pas de NaN
3. Le learning rate n'est pas trop élevé (réduire à `5e-6`)

### Modèle ne s'améliore pas

1. Vérifier les masks visuellement (voir section Monitoring)
2. S'assurer que les labels correspondent bien aux images
3. Augmenter le nombre d'epochs
4. Essayer un learning rate plus élevé (`5e-5`)

## 💾 Format de sauvegarde

Le modèle est sauvegardé au format SafeTensors (compatible Hugging Face) :

```
finetuned_model/
├── best_model/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── final_model/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

Vous pouvez partager ces modèles ou les charger dans d'autres scripts !

## 📚 Références

- Script original : `simple_inf.py`
- Architecture : EfficientViT pour Semantic Segmentation
- Repo Surya : https://github.com/VikParuchuri/surya
- Format YOLO : https://docs.ultralytics.com/datasets/detect/

---

**Bon fine-tuning ! 🚀**
