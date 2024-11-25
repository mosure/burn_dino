from PIL import Image
import torch
from torchvision import transforms
from featup.train_jbu_upsampler import JBUFeatUp
from featup.upsamplers import JBUStack


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

feature_upsampler: JBUFeatUp = torch.hub.load('mhamilton723/FeatUp', 'dinov2', use_norm=True).cuda().eval()
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
output = dinov2_vits14.forward_features(input)['x_norm_patchtokens']

upsampler: JBUStack = feature_upsampler.upsampler

upsampler.up1.range_temp = torch.nn.Parameter(upsampler.up1.range_temp.unsqueeze(0))
upsampler.up1.sigma_spatial = torch.nn.Parameter(upsampler.up1.sigma_spatial.unsqueeze(0))

upsampler.up2.range_temp = torch.nn.Parameter(upsampler.up2.range_temp.unsqueeze(0))
upsampler.up2.sigma_spatial = torch.nn.Parameter(upsampler.up2.sigma_spatial.unsqueeze(0))

upsampler.up3.range_temp = torch.nn.Parameter(upsampler.up3.range_temp.unsqueeze(0))
upsampler.up3.sigma_spatial = torch.nn.Parameter(upsampler.up3.sigma_spatial.unsqueeze(0))

upsampler.up4.range_temp = torch.nn.Parameter(upsampler.up4.range_temp.unsqueeze(0))
upsampler.up4.sigma_spatial = torch.nn.Parameter(upsampler.up4.sigma_spatial.unsqueeze(0))

torch.save(upsampler.state_dict(), './assets/models/upsampler.pth')
