import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from MammoNet.models.BaseModel import BaseModel


class VisionTransformerModel(nn.Module):
    """
    Vision Transformer model class.
    """

    def __init__(self, num_classes, model_name="google/vit-base-patch16-224-in21k"):
        super(VisionTransformerModel, self).__init__()
        self.name = "VisionTransformerModel"
        self.processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
        self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)

        # freeze backbone
        for param in self.model.vit.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(images.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.logits


class VisionTransformer(BaseModel):
    """
    Vision Transformer model class wrapper.
    """

    def __init__(self, num_classes=2, **kwargs):
        model = VisionTransformerModel(num_classes)
        super().__init__(model, num_classes, "ViT", lr=0.0005, epochs=10, **kwargs)
