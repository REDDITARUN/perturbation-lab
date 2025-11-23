from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


def get_device():
    """
    Select the best available device:
    - Apple MPS (Metal) on macOS
    - CUDA GPU
    - CPU as fallback
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Models:
    def resnet101(self):
        device = get_device()
        processor = AutoImageProcessor.from_pretrained(
            "microsoft/resnet-101", use_fast=True
        )
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-101")
        model.to(device).eval()
        return processor, model

    def efficientnet_b0(self):
        device = get_device()
        processor = AutoImageProcessor.from_pretrained(
            "google/efficientnet-b0", use_fast=True
        )
        model = AutoModelForImageClassification.from_pretrained(
            "google/efficientnet-b0"
        )
        model.to(device).eval()
        return processor, model

    def vit_base(self):
        device = get_device()
        processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", use_fast=True
        )
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        model.to(device).eval()
        return processor, model

    def swinv2(self):
        device = get_device()
        processor = AutoImageProcessor.from_pretrained(
            "microsoft/swinv2-base-patch4-window16-256", use_fast=True
        )
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-base-patch4-window16-256"
        )
        model.to(device).eval()
        return processor, model

    def model_list(self):
        """Return a list of (model_name, processor, model) tuples."""
        return [
            ("ResNet101", *self.resnet101()),
            ("EfficientNetB0", *self.efficientnet_b0()),
            ("ViTBase", *self.vit_base()),
            ("SwinV2", *self.swinv2()),
        ]
