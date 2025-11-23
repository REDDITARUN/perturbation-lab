# import necessary libraries
from models import Models
from PIL import Image
import os
from tabulate import tabulate
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic


def get_device():
    """
    Select the best available device for this script.
    Keeps logic in sync with models.py.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def custom_segmentation(image):
    """
    Custom segmentation function for LIME using SLIC.
    Produces fewer, larger superpixels for cleaner, less noisy explanations.
    """
    return slic(
        image,
        n_segments=50,
        compactness=20,
        sigma=1,
        start_label=0,
    )


explainer = lime_image.LimeImageExplainer(random_state=0)

# load all images in the sample directory
sample_directory = "images/"
image_list = sorted([f for f in os.listdir(sample_directory) if f.endswith(".jpg")])

# list of models
models = Models()
model_list = models.model_list()


def run_models(model_list):
    """Run all models on all images and return results grouped by image."""
    results_by_image = {}

    for model_name, processor, model in model_list:
        for image_filename in image_list:
            if image_filename not in results_by_image:
                results_by_image[image_filename] = {}

            image_path = os.path.join(sample_directory, image_filename)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class]
            results_by_image[image_filename][model_name] = predicted_label

    return results_by_image


def run_lime(processor, model, image):
    def predict_fn(images):
        inputs = processor(images=list(images), return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

        # Move back to CPU before converting to numpy for LIME
        return probs.detach().cpu().numpy()

    # Use the shared global explainer, providing our custom segmentation
    explanation = explainer.explain_instance(
        np.array(image),
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=500,  # match your notebook; increase (e.g. 500/1000) for better quality
        segmentation_fn=custom_segmentation,
    )

    return explanation


def save_explanation(explanation, image, model_name, image_name):
    """Save a clean, side‑by‑side LIME visualization with minimal, top‑K regions."""

    # --- 1. Build a positive‑only heatmap over superpixels, keep top‑K only ---
    segments = explanation.segments  # (H, W), superpixel ids
    top_label = explanation.top_labels[0]
    exp_list = explanation.local_exp[top_label]  # list[(seg_id, weight)]

    heatmap = np.zeros(segments.shape, dtype=float)
    K = 15  # number of most important positive regions to display
    pos_feats = [(seg_id, w) for seg_id, w in exp_list if w > 0]
    pos_feats = sorted(pos_feats, key=lambda x: x[1], reverse=True)[:K]

    for seg_id, weight in pos_feats:
        heatmap[segments == seg_id] = weight

    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val  # [0, 1]

    # --- 2. Backgrounds: original + dimmed version ---
    bg = np.array(image).astype(float) / 255.0
    bg_dim = np.clip(bg * 0.6, 0, 1)

    # Positive-only mask for a subtle contour (segment feel)
    temp, mask_pos = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    # --- 3. Figure: original (left) vs explanation (right) ---
    fig, (ax_orig, ax_expl) = plt.subplots(
        1,
        2,
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("white")

    # Left: original image
    ax_orig.imshow(bg)
    ax_orig.set_title("Original", loc="left", fontsize=9, color="#333")
    ax_orig.axis("off")

    # Right: overlay
    ax_expl.imshow(bg_dim)
    im = ax_expl.imshow(
        heatmap,
        cmap="GnBu",
        alpha=0.55,
        vmin=0.0,
        vmax=1.0,
    )
    try:
        ax_expl.contour(mask_pos, levels=[0.5], colors="white", linewidths=1.0)
    except Exception:
        pass
    ax_expl.set_title("Important regions", fontsize=9, color="#333")
    ax_expl.axis("off")

    # Figure title: model name only, prettified
    pretty_model = model_name.title()
    fig.suptitle(pretty_model, fontsize=11, color="#222", y=0.97)

    # --- 4. Small, minimal colorbar ---
    cbar = fig.colorbar(im, ax=ax_expl, fraction=0.03, pad=0.02)
    cbar.set_label("Relative importance", fontsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(7)

    # --- 5. Save with tight, clean margins (card‑like) ---
    output_dir = f"explanations/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    image_base_name = os.path.splitext(image_name)[0]
    output_path = f"{output_dir}/{image_base_name}.png"

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig)


def run_lime_explanations(model_list, image_list):
    """Run LIME explanations for all models on all images and save outputs."""
    print("\n" + "=" * 120)
    print("RUNNING LIME EXPLANATIONS")
    print("=" * 120)

    total_tasks = len(model_list) * len(image_list)
    current_task = 0

    for model_name, processor, model in model_list:
        print(f"\nProcessing model: {model_name}")
        for image_filename in image_list:
            current_task += 1
            print(
                f"  [{current_task}/{total_tasks}] Processing {image_filename}...",
                end=" ",
                flush=True,
            )

            try:
                image_path = os.path.join(sample_directory, image_filename)
                image = Image.open(image_path).convert("RGB")

                # Run LIME explanation
                explanation = run_lime(processor, model, image)

                # Save explanation
                save_explanation(explanation, image, model_name, image_filename)
                print("✓")

            except Exception as e:
                print(f"✗ Error: {str(e)}")

    print("\n" + "=" * 120)
    print("LIME EXPLANATIONS COMPLETE")
    print("=" * 120 + "\n")


def print_results(results_by_image):
    """Print results grouped by image for easy comparison."""
    # Extract model names dynamically from the first image's results
    if not results_by_image:
        print("No results to display.")
        return

    # Get model names from the first image (all images should have the same models)
    model_names = sorted(list(next(iter(results_by_image.values())).keys()))

    # Create table with image as rows and models as columns
    table_data = []
    for image_filename in sorted(results_by_image.keys()):
        row = {"Image": image_filename}
        for model_name in model_names:
            # Truncate long labels for better readability
            label = results_by_image[image_filename][model_name]
            if len(label) > 40:
                label = label[:37] + "..."
            row[model_name.replace("_", " ").title()] = label
        table_data.append(row)

    print("\n" + "=" * 120)
    print("MODEL PREDICTIONS COMPARISON (Grouped by Image)")
    print("=" * 120)
    print(tabulate(table_data, headers="keys", tablefmt="grid", stralign="left"))
    print()


if __name__ == "__main__":
    # Run models and get predictions
    print("Running models on all images...")
    results = run_models(model_list)
    print_results(results)

    # Run LIME explanations for all models on all images
    run_lime_explanations(model_list, image_list)
