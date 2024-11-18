from PIL import Image
from safetensors.torch import save_file
import torch
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

image = Image.open('./assets/images/dino_0.png')
input = transform(image).unsqueeze(0).cuda()

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()

local_vits = DinoVisionTransformer(518, 14, 3, 384, depth=12, num_heads=6)
local_vits.load_state_dict(dinov2_vits14.state_dict(), strict=True)
local_vits.cuda().eval()

output = local_vits.forward_features(input)['x_norm_patchtokens']

assert(torch.allclose(output, dinov2_vits14.forward_features(input)['x_norm_patchtokens'], atol=1e-3))

all_outputs = local_vits.debug_tensors
all_outputs.update({'input': input})


def debug_tensor(tensor, name):
    print('\n')
    print(f"{name}: {tensor.shape}")

    if isinstance(tensor, torch.Tensor):
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        median_val = tensor.median().item()

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"nan/inf in {name}")
    else:
        raise TypeError("Unsupported tensor type")

    print(f"min: {min_val}, max: {max_val}")
    print(f"mean: {mean_val}, median: {median_val}")


debug_tensor(output, 'dino_output')

for name, tensor in all_outputs.items():
    debug_tensor(tensor, name)

save_file({'output': output}, './assets/tensors/dino_0_small.st')
save_file(all_outputs, './assets/tensors/dino_0_small_debug.st')
