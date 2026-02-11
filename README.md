# 🗺️ Few-Shot Segmentation of Historical Maps via Linear Probing of Vision Foundation Models

## 📄 Abstract
As rich sources of history, maps provide crucial insights into historical changes, yet their diverse visual representations and limited annotated data pose significant challenges for automated processing. We propose a simple yet effective approach for few-shot segmentation of historical maps, leveraging the rich semantic embeddings of large vision foundation models combined with parameter-efficient fine-tuning. 🚀  

Our method outperforms the state-of-the-art on the Siegfried benchmark dataset, achieving relative improvements in mIoU of around **+20%** in the challenging 5-shot setting while requiring only **689k trainable parameters**—just **0.21%** of the total model size.

---

## ⚙️ Methodology
The approach follows a three-stage adaptation process:

1. **🔍 Extract**: Extract image embeddings from a foundation model such as SAM, DINOv2, or RADIO.  
2. **📐 Upscale**: Rescale embeddings back to the original image resolution.  
3. **🎯 Classify**: Apply a linear pixel-wise classifier to obtain segmentation masks.  

---

## 💻 Usage

### ⭐ Ours (RADIO-L + DoRA)
```bash
python run.py --class_name $class_name --base_model radio_l --nshots 5 --scale_factor 3 --adapter dora --exp_name reproduce_ours
```

### 📊 Baselines (e.g., UNet)
```bash
python run.py --class_name $class_name --base_model unet --nshots 5 --epochs 100 --scale_factor 1 --exp_name reproduce_baseline
```

### 📚 Citation
```bash
@InProceedings{few-shot2025sterzinger,
      author="Sterzinger, Rafael and Peer, Marco and Sablatnig, Robert",
      title="Few-Shot Segmentation of Historical Maps via Linear Probing of Vision Foundation Models",
      booktitle="Document Analysis and Recognition -- ICDAR 2025",
      year="2026",
      pages="425--442"
}

```
