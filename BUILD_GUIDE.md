# 🫁 Project 3: Chest X-Ray Segmentation — Build Guide + Deep Dive

---

## WHY THIS PROJECT IS DIFFERENT FROM PROJECTS 1 & 2

| | Project 1 | Project 2 | Project 3 |
|--|-----------|-----------|-----------|
| Task | Classification | Detection + Tracking | **Segmentation** |
| Output | Label ("happy") | Bounding boxes | **Pixel-level mask** |
| Model | Used pretrained | Used pretrained (YOLO) | **Build from scratch (U-Net)** |
| Tracking | W&B | W&B | **MLflow** (shows you know both) |
| Demo | Streamlit | Streamlit + FastAPI | **Gradio** (shows you know three) |
| Domain | General CV | Traffic/automotive | **Healthcare AI** |

This project completes your skill coverage. After this, you can say:
"I've done classification, detection, tracking, AND segmentation."

---

## DEEP DIVE: EVERY CONCEPT EXPLAINED

### 1. SEGMENTATION vs CLASSIFICATION vs DETECTION

**Classification:** "This X-ray shows pneumonia" → one label per image
**Detection:** "There's a nodule at (x,y,w,h)" → bounding box around findings
**Segmentation:** "These exact pixels are lung tissue" → every pixel gets a label

Segmentation is the hardest because the model must understand BOTH:
- WHAT is in the image (semantic understanding)
- WHERE exactly it is, pixel by pixel (spatial precision)

### 2. U-NET — WHY IT'S THE GOLD STANDARD FOR MEDICAL SEGMENTATION

**The problem U-Net solves:**

Regular CNNs (like your baseline CNN from Project 1) downsample images
through pooling layers: 256→128→64→32→16. This captures WHAT is in
the image but loses WHERE things are. You can't upsample back to 256×256
and get precise boundaries because the spatial information is gone.

**U-Net's solution — skip connections:**

```
ENCODER (what)                    DECODER (where)
256×256 ─── [64ch] ───────────────────────── [64ch] → 256×256 output
   │ pool                              ↑ upsample
128×128 ─── [128ch] ─────────────────── [128ch]
   │ pool                          ↑ upsample
64×64 ──── [256ch] ──────────────── [256ch]
   │ pool                      ↑ upsample
32×32 ──── [512ch] ────────────── [512ch]
   │ pool                  ↑ upsample
16×16 ──── [1024ch] BOTTLENECK
```

The horizontal arrows are **skip connections**. They pass the high-resolution
feature maps from the encoder directly to the decoder. The decoder combines:
- Upsampled features (coarse, knows WHAT) from below
- Skip features (detailed, knows WHERE) from the encoder

This is why U-Net gets precise boundaries — it never fully loses spatial info.

**Why build it from scratch:**
Every medical AI interview will ask "explain U-Net's architecture."
If you imported it from a library, you'd struggle. Building it block by block
means you understand every layer. Your `unet.py` is ~80 lines — clean,
readable, and fully yours.

**The DoubleConv block:**
```python
Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
```
Why TWO convolutions per block? One conv has a 3×3 receptive field.
Two consecutive convs have a 5×5 effective receptive field — seeing more
context — without the parameter cost of a 5×5 kernel.

**ConvTranspose2d (transposed convolution) for upsampling:**
The decoder needs to go from 16×16 back to 256×256. ConvTranspose2d is
a "learnable upsample" — instead of naive interpolation (bilinear/nearest),
the network learns HOW to upsample. This gives better results at boundaries.

### 3. DeepLabV3+ — THE COMPARISON MODEL

**Why include it:**
Your results table needs to compare models. U-Net vs DeepLabV3+ shows
you can evaluate different architectures objectively.

**How DeepLabV3+ differs from U-Net:**
- U-Net: encoder-decoder with skip connections
- DeepLabV3+: uses **atrous (dilated) convolutions** instead of pooling

**What atrous convolution is:**
Normal 3×3 conv sees 3×3 pixels. Atrous conv with dilation=2 sees 5×5 pixels
but only uses 9 weights (holes between them). Dilation=4 sees 9×9 pixels.
This captures multi-scale context without losing resolution.

**ASPP (Atrous Spatial Pyramid Pooling):**
DeepLabV3+ runs MULTIPLE dilated convolutions in parallel (dilation 1, 6, 12, 18)
and combines them. This captures objects at different scales simultaneously.

**For lung segmentation, U-Net usually wins because:**
- Lungs have clear, well-defined boundaries → skip connections excel
- Lungs don't vary much in scale → multi-scale ASPP isn't needed
- U-Net is simpler → less overfitting on 800 images

But showing you know both proves depth of knowledge.

### 4. CLAHE — WHY X-RAYS NEED SPECIAL PREPROCESSING

**The problem:**
X-rays have low dynamic range. Soft tissue (lungs, heart) and bone (ribs, spine)
have similar pixel intensities. A CNN trained on raw X-rays struggles to
distinguish these structures.

**What CLAHE does:**
Regular histogram equalization stretches pixel values globally — which amplifies
noise in uniform regions. CLAHE (Contrast Limited Adaptive Histogram Equalization)
works on small tiles (8×8 blocks). Each tile gets its own histogram equalization,
and the "contrast limit" prevents over-amplification of noise.

Result: lung boundaries become much sharper, ribs become more visible,
and the model can learn to distinguish structures more easily.

**Why clip_limit=2.0:**
Higher = more contrast enhancement but more noise. 2.0 is the standard
in medical imaging literature. You can experiment with 1.5-3.0.

### 5. LOSS FUNCTIONS — WHY DICE+BCE, NOT JUST BCE

**The class imbalance problem:**
In a 256×256 chest X-ray, roughly 40% of pixels are lung, 60% are background.
If the model predicts "background everywhere," it gets 60% pixel accuracy.
BCE (Binary Cross Entropy) treats every pixel equally, so it doesn't penalize
this behavior strongly enough.

**Dice Loss solves this:**
```
Dice = 2 × |prediction ∩ ground_truth| / (|prediction| + |ground_truth|)
```
If the model predicts zero lung pixels:
- prediction ∩ ground_truth = 0
- Dice = 0 / (0 + lots) = 0
- Loss = 1 - 0 = 1.0 (maximum loss!)

Dice directly penalizes under-prediction. It doesn't care about the background
pixels at all — only about how well the predicted lung region overlaps with
the true lung region.

**Why combine BCE + Dice:**
- Dice: good at getting the overall shape right (region-level gradient)
- BCE: good at getting boundaries right (pixel-level gradient)
- Together: best of both worlds. This is the standard in medical segmentation.

**Focal Loss (for comparison):**
Down-weights easy examples. Most pixels are easy (clearly lung or clearly
background). A few pixels are hard (at the boundary). Focal Loss focuses
the model's attention on these hard boundary pixels. Less commonly used
than Dice+BCE for lung segmentation but worth comparing.

### 6. MEDICAL METRICS — WHY STANDARD ML METRICS FAIL

**Accuracy is meaningless:**
60% of pixels are background. Predict all background → 60% accuracy. Useless.

**Dice Score (what everyone reports):**
```
Dice = 2|A∩B| / (|A|+|B|)
```
0 = no overlap, 1 = perfect overlap. This is THE metric for segmentation.
In lung segmentation, state-of-the-art Dice is 0.96-0.98.

**IoU (Jaccard):**
```
IoU = |A∩B| / |A∪B|
```
Stricter than Dice (Dice ≥ IoU always). Dice of 0.95 ≈ IoU of 0.90.

**Sensitivity (Recall):**
Of all actual lung pixels, how many did we find? Critical because
MISSING lung tissue in a segmentation could lead to incorrect measurements.

**Specificity:**
Of all background pixels, how many did we correctly identify?
High specificity = low false positive rate = we're not hallucinating lungs.

### 7. MLflow — WHY AND HOW

**Why MLflow instead of W&B again:**
- You already used W&B in Project 1 → shows you know ONE tool
- Using MLflow here → shows you know BOTH major tracking platforms
- MLflow is open-source, self-hosted → many companies prefer it
- MLflow has a model registry → version your models, stage them
  (staging → production), track lineage

**How it works:**
```python
mlflow.set_experiment("chest-xray-segmentation")
with mlflow.start_run(run_name="unet-scratch"):
    mlflow.log_params({"lr": 0.001, "loss": "dice_bce"})  # hyperparams
    mlflow.log_metrics({"val_dice": 0.94}, step=epoch)     # per-epoch metrics
    mlflow.pytorch.log_model(model, "best_model")          # save model artifact
```

Access the MLflow UI:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 8. ELASTIC TRANSFORM — THE KEY MEDICAL AUGMENTATION

**Why it matters:**
Patient anatomy varies. Ribs curve differently, diaphragm sits higher/lower,
heart is different sizes. ElasticTransform simulates these anatomical variations
by warping the image grid, creating realistic deformations that wouldn't
happen with simple rotation/scaling.

**Why NOT vertical flip:**
Lungs don't appear upside down in real X-rays. The machine always captures
with the patient standing upright. Vertical flip would create unrealistic
training data. Horizontal flip IS valid because left-right mirroring is
anatomically plausible (though asymmetric — left lung has 2 lobes, right has 3).

---

## DAY-BY-DAY BUILD PLAN

### DAY 1-3: Setup + Data + Baseline

```bash
tar xzf chest-xray-segmentation.tar.gz
cd chest-xray-segmentation
conda create -n xray-seg python=3.10 -y
conda activate xray-seg
pip install -r requirements.txt

# Download data
python scripts/download_data.py
```

**Claude Code prompt — organize dataset:**
```
The chest X-ray dataset I downloaded needs to be organized into this structure:
data/chest_xray/
  train/images/  train/masks/
  val/images/    val/masks/
  test/images/   test/masks/

Split: 80% train, 10% val, 10% test.
The Montgomery dataset has separate left/right lung masks — combine them 
into single masks using src/data/preprocessing.py combine_lung_masks().
Write a script at scripts/prepare_splits.py that does this automatically.
Ensure every image has a matching mask with the same filename stem.
```

```bash
# Verify U-Net works
python src/models/unet.py

# Train U-Net
python src/models/train.py --config configs/unet.yaml

git init && git add . && git commit -m "feat: U-Net training on chest X-ray data"
```

### DAY 4-7: DeepLabV3+ + Comparison

```bash
python src/models/train.py --config configs/deeplabv3.yaml

# View experiments in MLflow
mlflow ui --port 5000
```

**Claude Code prompt — evaluation & visualization:**
```
Create src/evaluation/evaluate.py that:
1. Loads a trained model checkpoint
2. Runs it on the entire test set
3. Computes per-image Dice, IoU, sensitivity, specificity
4. Generates:
   - A classification report table (mean ± std for each metric)
   - Prediction overlay visualizations for 10 random test images 
     (original X-ray | ground truth mask | predicted mask | overlay)
   - A "failure cases" analysis: find the 5 worst-performing images 
     by Dice score, visualize them, and analyze WHY the model failed
   - A Dice score histogram showing distribution across test set
5. Saves all visualizations to assets/evaluation/
6. Logs everything to MLflow as artifacts
```

**Claude Code prompt — experiment comparison:**
```
Create a comparison script at scripts/compare_models.py that:
1. Loads both U-Net and DeepLabV3+ best checkpoints
2. Runs both on the same test set
3. Creates a side-by-side comparison table (Dice, IoU, sensitivity, 
   specificity, inference time, parameter count)
4. Generates side-by-side prediction visualizations (same image, 
   both models' outputs)
5. Saves comparison to assets/model_comparison.png
```

```bash
git add . && git commit -m "feat: DeepLabV3+ training + model comparison"
```

### DAY 8-10: Polish + Gradio App + Deploy

```bash
# Test Gradio app
python src/app/app.py
# Opens a browser with the demo
```

**Claude Code prompt — enhance Gradio app:**
```
Enhance the Gradio app at src/app/app.py with:
1. Model selection dropdown: choose between U-Net and DeepLabV3+
2. A threshold slider (0.1 to 0.9) for mask binarization
3. Side-by-side comparison: show original, mask, overlay, and probability map
4. Display Dice score if a ground truth mask is also uploaded
5. Add sample X-ray images as examples (from the test set)
```

**Claude Code prompt — deploy to HuggingFace:**
```
Deploy this project to HuggingFace Spaces as a Gradio app.
Create the proper README with YAML frontmatter.
Include the trained model weights in the repo (or download them at startup).
The app entry point is src/app/app.py.
```

**Claude Code prompt — tests + CI:**
```
Write comprehensive tests:
1. test_unet.py: test output shape, skip connections, different input sizes
2. test_metrics.py: test dice/iou/sensitivity/specificity with known values
3. test_losses.py: test DiceLoss, DiceBCELoss, FocalLoss with synthetic data
4. test_dataset.py: test transforms, mask processing, CLAHE

Then set up GitHub Actions CI at .github/workflows/ci.yml:
- On push: run pytest + flake8
- Cache pip dependencies
```

```bash
git add . && git commit -m "feat: Gradio app, tests, CI/CD, HuggingFace deployment"
git remote add origin https://github.com/krishna8399/chest-xray-segmentation.git
git push -u origin main
```

---

## INTERVIEW QUESTIONS

**1. "Why U-Net over other architectures?"**
→ U-Net excels with small datasets (we only have 800 images). Skip connections
preserve spatial detail for precise lung boundaries. It's also the most
well-understood architecture in medical imaging — every radiologist and ML
engineer knows it. I also compared with DeepLabV3+ to show the tradeoff.

**2. "Why Dice Loss instead of CrossEntropy?"**
→ Class imbalance. 60% of pixels are background. CrossEntropy gives equal
weight to every pixel, so predicting all-background gets 60% accuracy.
Dice Loss directly optimizes the overlap metric — it doesn't care about
background pixels, only about how well the predicted lung matches ground truth.

**3. "How would you handle multi-class segmentation?"**
→ Change out_channels from 1 to N (number of classes). Replace sigmoid with
softmax. Replace binary Dice with multi-class Dice (compute per-class, average).
Replace BCEWithLogitsLoss with CrossEntropyLoss. The architecture stays the same.

**4. "What are the limitations of your approach?"**
→ Small dataset (800 images) limits generalization to different X-ray machines
and patient populations. The model was trained on Montgomery (US) + Shenzhen
(China) data — it may underperform on X-rays from different hospitals with
different equipment. Solution: more data, data augmentation, and domain adaptation.

**5. "How would you deploy this in a hospital?"**
→ DICOM integration (hospitals use DICOM format, not PNG). Inference server
behind the hospital firewall (can't send patient data to cloud — GDPR/HIPAA).
Model versioning with MLflow model registry. Human-in-the-loop: model suggests
segmentation, radiologist verifies and corrects. Continuous learning from corrections.

**6. "Why MLflow and not W&B?"**
→ I use both. W&B in Project 1, MLflow here. MLflow is open-source and self-hosted,
which is important for healthcare where data can't leave the hospital network.
MLflow's model registry also supports staging/production model promotion, which
is critical for regulated medical devices.

---

## YOU vs CLAUDE CODE

| YOU decide | CLAUDE CODE implements |
|-----------|----------------------|
| Dataset split strategy | Split script with stratification |
| Which augmentations make medical sense | Transform pipeline code |
| Analyze failure cases (why model fails) | Visualization code for failures |
| Compare U-Net vs DeepLabV3+ results | Side-by-side comparison script |
| Choose threshold for clinical use | Threshold sweep + ROC curve code |
| Write "What I Learned" from real experience | README formatting |
| Deploy to HuggingFace | Deployment config and restructuring |
