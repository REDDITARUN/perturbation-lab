## perturbation-lab

Explore how LIME interprets CNNs and Transformers. Segment images into superpixels, perturb them, fit a local surrogate, and see exactly what the model is looking at. It runs several pretrained vision models from Hugging Face on local images and saves clean, side‑by‑side explanation plots.

### What it does

- **Models** (see `models.py`):
  - `ResNet101` – `microsoft/resnet-101`
  - `EfficientNetB0` – `google/efficientnet-b0`
  - `ViTBase` – `google/vit-base-patch16-224`
  - `SwinV2` – `microsoft/swinv2-base-patch4-window16-256`
- **Inputs**: all `.jpg` files in the `images/` folder.
- **Outputs**:
  - A comparison table of model predictions per image (printed to the terminal).
  - One explanation PNG per `(model, image)` saved under `explanations/{ModelName}/`.

### How to run

- **Setup (optional but recommended)**

```bash
python -m venv env
source env/bin/activate  # macOS / Linux
# .\env\Scripts\activate  # Windows
pip install -r requiremnets.txt
```

- **Run the experiment**

```bash
python run.py
```

Images in the `images/` folder are processed by all models, and LIME explanations are saved under `explanations/{ModelName}/`.

### How it works (short version)

- **Device selection**: automatically uses Apple **MPS**, then **CUDA**, then **CPU** (see `get_device()` in `models.py` / `run.py`).
- **Prediction**:
  - Hugging Face `AutoImageProcessor` handles resizing / normalization.
  - Models run on the selected device; outputs are converted back to NumPy for LIME.
- **LIME**:
  - Uses a custom SLIC segmentation with a small number of superpixels for cleaner regions.
  - Highlights only the **top‑K positive** regions for a given prediction.
  - Visualizes original vs. explanation side‑by‑side with a minimal colormap and small colorbar.



### Article

- **Long‑form write‑up**: *(coming soon – link to article will go here)*  

### References

[1] M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why Should I Trust You?’: Explaining the Predictions of Any Classifier,” arXiv preprint arXiv:1602.04938, 2016.  
[2] M. T. Ribeiro, “lime: Explaining the predictions of any machine learning classifier,” GitHub repository and documentation.  


