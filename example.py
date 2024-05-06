
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image

from augmentations import generate_hist_augs, opts, mean_domains, std_domains
from histaugan.model import MD_multi

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# add correct path to directory with sample tiles (5 images per domain, hence in total 25 tiles)
tiles_dir = 'sample_tiles/'
tile_names = os.listdir(tiles_dir)
tile_names.sort()
if '.ipynb_checkpoints' in tile_names:
    tile_names.pop(0)
print(tile_names)
# plot all images
# rows, columns, img_size = 5, 5, 2

# plt.figure(figsize=(columns * img_size, rows * img_size))
# for i in range(rows * columns):
#     img = torch.load(tiles_dir + tile_names[i])
#
#     plt.subplot(rows, columns, (i % columns) * columns + (i // columns) + 1)
#     plt.imshow(img.permute(1, 2, 0))
#     if i % 5 == 0:
#         plt.title(f'domain {i // 5}', fontsize=18)
#     plt.axis('off')
# plt.tight_layout()
model = MD_multi(opts)
model.resume(opts.resume, train=False)
model.to(device)
model.eval()
print('--- model loaded ---')
# choose a sample tile
domain = np.random.randint(2)
img_id = np.random.randint(2)
rows, columns, img_size = 1, 2, 2
plt.figure(figsize=(columns * img_size, rows * img_size))

img = torch.load(tiles_dir + tile_names[domain * 5 + img_id]).to(device)
plt.subplot(rows, columns, 1)
plt.imshow(img.permute(1, 2, 0).cpu())
plt.title(f'original (domain {domain})')
plt.axis('off')

z_content = model.enc_c(img.sub(0.5).mul(2).unsqueeze(0))

for i in range(rows * columns - 1):
    out = generate_hist_augs(img, domain, model, z_content, same_attribute=False, new_domain=i, stats=(mean_domains, std_domains), device=device)

    plt.subplot(rows, columns, i + 2)
    plt.imshow(out.add(1).div(2).permute(1, 2, 0).cpu())
    plt.title(f'synthetic (domain {i})')
    plt.axis('off')

plt.tight_layout()
plt.show()