import math
from PIL import Image
from safetensors.torch import save_file
import torch
from torch_pca import PCA
from torchvision import transforms

from dinov2.models.vision_transformer import DinoVisionTransformer

transform = transforms.Compose([
    transforms.Resize(520, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

images = [
    './assets/images/dino_0.png',
    './assets/images/dino_1.png',
    './assets/images/dino_2.png',
    './assets/images/dino_3.png',
]

inputs = []
for image_path in images:
    image = Image.open(image_path)
    input = transform(image).unsqueeze(0).cuda()
    inputs.append(input)

input = torch.cat(inputs, dim=0)


dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()

local_vits = DinoVisionTransformer(518, 14, 3, 384, depth=12, num_heads=6)
local_vits.load_state_dict(dinov2_vits14.state_dict(), strict=True)
local_vits.cuda().eval()

output = local_vits.forward_features(input)['x_norm_patchtokens']
collapsed_output = output.reshape(-1, output.shape[-1])

pca = PCA(n_components=3)
pca_fg = PCA(n_components=3)

pca.fit(collapsed_output)
pca_features = pca.transform(collapsed_output)

pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                     (pca_features[:, 0].max() - pca_features[:, 0].min())


# TODO: export pca_features for testing
# TODO: export components and mean
components = pca.components_
mean = pca.mean_

print('components:', components.shape)
print('mean:', mean.shape)

save_file(
    {
        'components': components,
        'mean': mean,
    },
    './assets/tensors/dino_pca.st',
)


pca_features_bg = pca_features[:, 0] > 0.35
pca_features_fg = ~pca_features_bg

fg_features = collapsed_output[pca_features_fg]
pca_fg.fit(fg_features)
pca_features_left = pca_fg.transform(fg_features)

# save_file(
#     {
#         'fg_features': fg_features,
#         'components': pca_fg.components_,
#         'mean': pca_fg.mean_,
#     },
#     './assets/tensors/dino_fg_pca.st',
# )


for i in range(3):
    pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

pca_features_rgb = pca_features.clone()
pca_features_rgb[pca_features_bg] = 0
pca_features_rgb[pca_features_fg] = pca_features_left
pca_features_rgb = pca_features_rgb.reshape(input.shape[0], -1, 3)
pca_features_rgb = pca_features_rgb.reshape(
    pca_features_rgb.shape[0],
    math.isqrt(pca_features_rgb.shape[1]),
    math.isqrt(pca_features_rgb.shape[1]),
    3,
)

pca_features_rgb = pca_features_rgb.detach().cpu().numpy()

for i in range(pca_features_rgb.shape[0]):
    image = transforms.ToPILImage()(pca_features_rgb[i])
    image = image.resize((518, 518), resample=Image.LANCZOS)

    image.save(f'./assets/pca/dino_{i}_pca.png')
