"""Research classifiers with a single binary logit."""
from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


def pretrained_url(architecture: str) -> str:
    if architecture == "efficientnet_b0":
        return EfficientNet_B0_Weights.IMAGENET1K_V1.url
    if architecture == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2.url
    raise ValueError(f"Unknown architecture: {architecture}")


def build_model(architecture: str, fine_tuning: str, pretrained: bool = True) -> nn.Module:
    """Create an ImageNet initialization or an explicitly untrained smoke-test model.

    'partial' unfreezes the last three EfficientNet feature blocks or ResNet layer4.
    'full' unfreezes all parameters; 'frozen' trains only the classification head.
    """
    if fine_tuning not in {"frozen", "partial", "full"}:
        raise ValueError(f"Unknown fine-tuning mode: {fine_tuning}")
    if architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        for parameter in model.features.parameters():
            parameter.requires_grad = fine_tuning == "full"
        if fine_tuning == "partial":
            for parameter in model.features[-3:].parameters():
                parameter.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif architecture == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        for parameter in model.parameters():
            parameter.requires_grad = fine_tuning == "full"
        if fine_tuning == "partial":
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def set_training_mode(model: nn.Module) -> None:
    """Avoid changing running statistics in frozen batch-normalization modules."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
            module.eval()
