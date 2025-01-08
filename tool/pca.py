import math
from PIL import Image
from safetensors.torch import save_file
import torch
import torch.nn as nn
from torch_pca import PCA
from torchvision import transforms


class PcaModel(nn.Module):
    def __init__(
        self,
        components,
        mean,
    ) -> None:
        super().__init__()

        self.components = nn.Parameter(components)
        self.mean = nn.Parameter(mean)

    def forward(self, x):
        return x


transform = transforms.Compose([
    transforms.Resize(520, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

prefix = 'face'
images = [
    f'./assets/images/{prefix}_0.webp',
    f'./assets/images/{prefix}_1.webp',
    f'./assets/images/{prefix}_2.webp',
    f'./assets/images/{prefix}_3.webp',
    f'./assets/images/{prefix}_4.webp',
    f'./assets/images/{prefix}_5.webp',
]

inputs = []
for image_path in images:
    image = Image.open(image_path).convert('RGB')
    input = transform(image).unsqueeze(0).cuda()
    inputs.append(input)

input = torch.cat(inputs, dim=0)


dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
output = dinov2_vits14.forward_features(input)['x_norm_patchtokens']
collapsed_output = output.reshape(-1, output.shape[-1])

pca = PCA(n_components=3)
pca_fg = PCA(n_components=3)

pca.fit(collapsed_output)
pca_features = pca.transform(collapsed_output)

pca_features_pre_min_max = pca_features.clone()


for i in range(3):
    pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())



components = pca.components_
mean = pca.mean_

print('components:', components.shape)
print('mean:', mean.shape)
print('pca_features:', pca_features.shape)
print('pca_features_pre_min_max:', pca_features_pre_min_max.shape)

# TODO: export expected transform input shape
# save_file(
#     {
#         'components': components,
#         'dino_output': collapsed_output,
#         'input': input,
#         'mean': mean,
#         'pca_features': pca_features,
#         'pca_features_pre_min_max': pca_features_pre_min_max,
#     },
#     './assets/tensors/dino_pca.st',
# )
torch.save(PcaModel(components, mean).state_dict(), f'./assets/models/{prefix}_pca.pth')


pca_features_bg = pca_features[:, 0] > 0.5
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
torch.save(PcaModel(pca_fg.components_, pca_fg.mean_).state_dict(), f'./assets/models/{prefix}_fg_pca.pth')


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
    image = image.resize((1024, 1024), resample=Image.LANCZOS)

    image.save(f'./assets/pca/{prefix}_{i}_pca.png')
