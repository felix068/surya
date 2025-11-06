# 🔴 Troubleshooting Fine-tuning Issues

## Problème : Le modèle fine-tuné est PIRE que l'original

### Symptômes observés :

1. ✅ Training fonctionne (loss descend un peu : 1.26 → 1.20)
2. ❌ Loss finale trop haute (1.2 au lieu de <0.5)
3. ❌ Modèle file size double (70MB → 150MB)
4. ❌ **Le pire** : Modèle détecte des ZONES au lieu de LIGNES

---

## 🔍 Diagnostic

### Problème 1 : Taille du fichier (70MB → 150MB)

**Cause** : Le modèle a été sauvegardé en **float32** au lieu de **float16**

```python
73 MB (fp16) × 2 = 146 MB (fp32)
```

**Solution** : ✅ Corrigé dans la dernière version de `simple_fine.py`

Le modèle est maintenant chargé et sauvegardé en fp16 sur GPU.

---

### Problème 2 : Loss stagne à 1.2

**Causes possibles** :

1. **Dataset trop petit** : 248 images << 1000 minimum
2. **Learning rate trop faible** : 1e-5 ne permet pas d'apprendre assez vite
3. **Tout le modèle fine-tuné** : Avec peu de données, ça détruit les features pré-entraînées

**Solution** : ✅ Corrigé dans la dernière version

- Encoder **freezé** automatiquement si <1000 images
- Seulement le decoder est fine-tuné (moins de params à apprendre)

---

### Problème 3 : Détecte des ZONES au lieu de LIGNES 🚨

**C'est le plus GRAVE !** Cela signifie que :

#### Hypothèse A : Tes annotations sont INCORRECTES

Tes bboxes YOLO annotent-elles vraiment des **lignes individuelles** ou des **paragraphes/zones** ?

```
❌ MAUVAIS (annotation de ZONE) :
┌─────────────────────────────┐
│ This is line 1              │
│ This is line 2              │  ← UNE SEULE bbox pour tout !
│ This is line 3              │
└─────────────────────────────┘
Label YOLO : 0 0.5 0.3 0.8 0.15

✅ BON (annotation de LIGNES séparées) :
┌─────────────────────────────┐  ← Bbox 1
│ This is line 1              │
└─────────────────────────────┘
Label YOLO : 0 0.5 0.1 0.8 0.05

┌─────────────────────────────┐  ← Bbox 2
│ This is line 2              │
└─────────────────────────────┘
Label YOLO : 0 0.5 0.2 0.8 0.05

┌─────────────────────────────┐  ← Bbox 3
│ This is line 3              │
└─────────────────────────────┘
Label YOLO : 0 0.5 0.3 0.8 0.05
```

#### Comment vérifier ?

**Exécute ce script** :

```bash
python analyze_yolo_annotations.py
```

Regarde :
- `bbox_analysis.png` : Histogramme des aspect ratios
- `bbox_samples.png` : Échantillons avec bboxes dessinés

**Interprétation des aspect ratios** :

| Aspect Ratio (width/height) | Type | Verdict |
|------------------------------|------|---------|
| **> 15:1** | Ligne de texte | ✅ BON |
| **10:1 à 15:1** | Ligne courte | ✅ OK |
| **3:1 à 10:1** | Ambigü | ⚠️ Vérifier visuellement |
| **< 3:1** | Zone/Paragraphe | ❌ MAUVAIS |

**Exemple** :

```
Bbox : 800px width × 40px height
Aspect ratio = 800/40 = 20:1 ✅ C'est une LIGNE

Bbox : 600px width × 200px height
Aspect ratio = 600/200 = 3:1 ❌ C'est une ZONE
```

#### Hypothèse B : Overfitting catastrophique

Avec seulement **248 images**, le modèle :
1. A complètement overfit sur ces 248 images
2. A **oublié** ce qu'il avait appris sur des centaines de milliers d'images
3. A appris un nouveau pattern (zones) qui écrase l'ancien (lignes)

**C'est pour ça que freezer l'encoder est CRITIQUE !**

---

## ✅ Solutions

### Solution 1 : Re-fine-tuner avec la version corrigée

La nouvelle version de `simple_fine.py` inclut :

```python
# 1. Sauvegarde en fp16 (taille correcte)
dtype = torch.float16

# 2. Freeze automatique de l'encoder si <1000 images
if len(dataset) < 1000:
    for name, param in model.named_parameters():
        if "decode_head" not in name:
            param.requires_grad = False

# Résultat :
# Au lieu de fine-tuner 38M params → seulement ~5M params (decoder)
```

### Solution 2 : Augmenter le learning rate

Pour les petits datasets, modifie dans `simple_fine.py` :

```python
class Config:
    LEARNING_RATE = 1e-4  # Au lieu de 1e-5 (10x plus !)
    EPOCHS = 50           # Plus d'epochs
```

### Solution 3 : Vérifier tes annotations AVANT de fine-tuner

**OBLIGATOIRE** :

```bash
# 1. Analyser les bboxes
python analyze_yolo_annotations.py

# 2. Regarder bbox_samples.png
# → Chaque bbox doit être UNE LIGNE, pas un paragraphe

# 3. Si tes bboxes sont des zones, RE-ANNOTER correctement !
```

### Solution 4 : Utiliser le modèle original

Si ton dataset est correct et que Surya original marche déjà bien :

**Ne fine-tune PAS !** 🛑

Avec 248 images, tu risques de **dégrader** le modèle plutôt que de l'améliorer.

Le fine-tuning n'est utile que si :
- Tu as **>1000 images** minimum
- Ton domaine est **très spécifique** (police rare, layout unique, etc.)
- Le modèle original **échoue** sur tes données

Sinon, utilise directement le modèle pré-entraîné !

---

## 🎯 Workflow recommandé

### Étape 1 : Vérifier les annotations

```bash
python analyze_yolo_annotations.py
```

**Critères de validation** :
- ✅ Aspect ratio médian > 10:1
- ✅ Hauteur médiane : 20-60px
- ✅ Visuellement : 1 bbox = 1 ligne de texte

Si **NON** → Re-annoter correctement !

### Étape 2 : Évaluer le modèle original

```bash
python simple_inf.py
```

Si le modèle original marche **bien** → **NE PAS fine-tuner !**

### Étape 3 : Si vraiment nécessaire, fine-tuner avec précautions

```bash
# Config recommandée pour 248 images :
LEARNING_RATE = 1e-4  # Plus élevé
EPOCHS = 50           # Plus long
BATCH_SIZE = 4

# L'encoder sera automatiquement freezé
python simple_fine.py
```

### Étape 4 : Comparer AVANT/APRÈS

```bash
# Test avec modèle original
MODEL_CHECKPOINT = "s3://text_detection/2025_05_07"
python simple_inf.py

# Test avec modèle fine-tuné
MODEL_CHECKPOINT = "./finetuned_model/best_model"
python simple_inf.py

# Le fine-tuné doit être MEILLEUR ou équivalent
# Si PIRE → Revenir au modèle original !
```

---

## 📊 Métriques attendues

### Bon fine-tuning :

```
Loss finale : 0.2 - 0.5
Taille fichier : ~73 MB (fp16)
Résultat : Meilleur ou égal au modèle original
Trainable params : ~5M (decoder seulement) si <1000 images
```

### Mauvais fine-tuning (ton cas) :

```
Loss finale : 1.2 ❌
Taille fichier : 150 MB ❌ (maintenant corrigé)
Résultat : PIRE que l'original ❌
Trainable params : 38M (tout le modèle) ❌ (maintenant corrigé)
```

---

## 🔧 Corrections appliquées

Dans la dernière version de `simple_fine.py` :

1. ✅ **Dtype fix** : Modèle chargé en fp16 sur GPU
2. ✅ **Auto-freeze** : Encoder freezé si <1000 images
3. ✅ **Interpolation** : Logits interpolés pour matcher mask size

---

## 💡 Conclusion

Avec **248 images seulement** :

1. **Vérifie d'abord** que tes annotations sont des LIGNES, pas des ZONES
2. **Teste le modèle original** - il marche peut-être déjà parfaitement
3. Si tu dois fine-tuner :
   - Utilise la version corrigée de `simple_fine.py`
   - Augmente le learning rate à 1e-4
   - Freeze l'encoder (automatique maintenant)
   - Compare AVANT/APRÈS obligatoirement

4. **Si le fine-tuné est PIRE** : Reviens au modèle original !

Le fine-tuning n'est pas magique. Avec peu de données, il peut faire **plus de mal que de bien**.

---

**Questions à te poser** :

1. Mes annotations sont-elles vraiment des LIGNES individuelles ?
2. Le modèle original échoue-t-il vraiment sur mes données ?
3. Ai-je comparé les performances AVANT/APRÈS ?
4. Est-ce que 248 images suffisent pour mon cas d'usage ?

Si doute → Utilise le modèle original sans fine-tuning ! 🎯
