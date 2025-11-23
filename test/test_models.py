import pytest
from models import Models


# Automatically discover all model methods at module level
def get_all_model_methods():
    """Get all public methods from Models class."""
    return [
        name
        for name in dir(Models())
        if not name.startswith("_") and callable(getattr(Models(), name))
    ]


@pytest.mark.parametrize("model_method_name", get_all_model_methods())
def test_model_loads(model_method_name):
    """Test that each model loads successfully."""
    models_instance = Models()
    method = getattr(models_instance, model_method_name)
    processor, model = method()

    # Check processor is not None and has processor-like functionality
    assert processor is not None, f"{model_method_name} processor should not be None"
    assert hasattr(processor, "__call__"), (
        f"{model_method_name} processor should be callable"
    )
    assert hasattr(processor, "preprocess") or hasattr(
        processor, "image_processor_type"
    ), f"{model_method_name} processor should have processor attributes"

    # Check model is not None and has model-like functionality
    assert model is not None, f"{model_method_name} model should not be None"
    assert hasattr(model, "forward"), (
        f"{model_method_name} model should have forward method"
    )
    assert hasattr(model, "config"), (
        f"{model_method_name} model should have config attribute"
    )
    assert hasattr(model, "eval"), (
        f"{model_method_name} model should be a PyTorch model"
    )
