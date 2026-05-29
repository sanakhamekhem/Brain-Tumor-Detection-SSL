import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Chemin vers votre modèle fine-tuné
model_path = '/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/code/MAE/resultatbase2448patch8/checkpoint-49.pth'

# Charger le modèle fine-tuné ViT
model = ViTForImageClassification.from_pretrained('vit_base_patch8_224')
model.load_state_dict(torch.load(model_path))
model.eval()



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self.register_hooks()

    def save_gradient(self, grad):
        self.gradients = grad

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def generate_cam(self, input_image, target_class):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_image)

        target = output.logits[:, target_class]
        target.backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def __del__(self):
        for hook in self.hooks:
            hook.remove()





# Charger une image d'exemple
img_path = '/mnt/8842af1e-476b-40b4-ad2c-c1ee3134b2c1/Mastere_Bidine/datasets/data_augmenter80test20%/val/yes/Y1.jpg'
img = Image.open(img_path)

# Prétraitement de l'image avec ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('vit_base_patch8_224')
inputs = feature_extractor(images=img, return_tensors="pt")

# Supposons que 'target_class' soit la classe que vous souhaitez visualiser (0 ou 1 pour la classification binaire)
target_class = 1  # Adapter selon votre cas

# Supposons que 'model.transformer.encoder.layers[-1].output' soit la dernière couche de transformation
target_layer = model.transformer.encoder.layers[-1].output

# Créer une instance de GradCAM
grad_cam = GradCAM(model, target_layer)

# Générer la carte de chaleur Grad-CAM
cam = grad_cam.generate_cam(inputs['pixel_values'], target_class)

# Visualisation du CAM superposé sur l'image originale
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
final_image = heatmap + np.float32(inputs['pixel_values'].squeeze().permute(1, 2, 0).cpu().numpy())
final_image = final_image / np.max(final_image)

plt.imshow(final_image)
plt.axis('off')
plt.show()
