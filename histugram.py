import glob
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
from tqdm import tqdm
save_data = False
output_to_one = "/data/results_lr_30_d_iter_2/val/0/"
output_to_zero = "/data/results_lr_30_d_iter_2/val/1/"
data = "/data/data/proc-images/"
domain_folders = os.listdir(data)
domain1_path= os.path.join(data, domain_folders[0])
domain2_path = os.path.join(data, domain_folders[1])

pickle_trainA_path = 'trainA_.pkl'  # Path to the pickle file
pickle_trainB_path = 'trainB.pkl'  # Path to the pickle file
picke_results_A2B =  'resultsA2B.pkl'
picke_results_B2A =  'resultsB2A.pkl'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def calculate_histogram_batch(images_tensor):
    batch_hist_r = torch.zeros(256).to(images_tensor.device)
    batch_hist_g = torch.zeros(256).to(images_tensor.device)
    batch_hist_b = torch.zeros(256).to(images_tensor.device)

    for image_tensor in images_tensor:
        hist_r = torch.histc(image_tensor[0], bins=256, min=0, max=255)
        hist_g = torch.histc(image_tensor[1], bins=256, min=0, max=255)
        hist_b = torch.histc(image_tensor[2], bins=256, min=0, max=255)
        batch_hist_r += hist_r
        batch_hist_g += hist_g
        batch_hist_b += hist_b

    return batch_hist_r, batch_hist_g, batch_hist_b


def process_folder(folder_path, batch_size=32):
    image_files = glob.glob(folder_path + '/*.*')
    hist_r = torch.zeros(256).to(device)
    hist_g = torch.zeros(256).to(device)
    hist_b = torch.zeros(256).to(device)

    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
        batch_files = image_files[i:i + batch_size]
        images_tensor = torch.stack(
            [torch.tensor(cv2.imread(file).transpose(2, 0, 1)).float().to(device) for file in batch_files])
        batch_hist_r, batch_hist_g, batch_hist_b = calculate_histogram_batch(images_tensor)
        hist_r += batch_hist_r
        hist_g += batch_hist_g
        hist_b += batch_hist_b

    return hist_r, hist_g, hist_b


# Process images one by one for both domains and accumulate histogram data
if not os.path.exists(pickle_trainA_path):
    with open(pickle_trainA_path, 'wb') as f:
        pickle.dump(process_folder(domain1_path), f)

if not os.path.exists(pickle_trainB_path):
    with open(pickle_trainB_path, 'wb') as f:
        pickle.dump(process_folder(domain2_path), f)

if not os.path.exists(picke_results_A2B):
    with open(picke_results_A2B, 'wb') as f:
        pickle.dump(process_folder(output_to_one), f)

if not os.path.exists(picke_results_B2A):
    with open(picke_results_B2A, 'wb') as f:
        pickle.dump(process_folder(output_to_zero), f)

with open(pickle_trainA_path, 'rb') as f:
    domain1_r_hist, domain1_g_hist, domain1_b_hist = pickle.load(f)
with open(pickle_trainB_path, 'rb') as f:
    domain2_r_hist, domain2_g_hist, domain2_b_hist = pickle.load(f)
with open(picke_results_A2B, 'rb') as f:
    output_to_one_r_hist, output_to_one_g_hist, output_to_one_b_hist = pickle.load(f)
with open(picke_results_B2A, 'rb') as f:
    output_to_zero_r_hist, output_to_zero_g_hist, output_to_zero_b_hist = pickle.load(f)



# Plot histograms
def plot_histogram(hist1, channel_name, color1 ):
    plt.figure()
    plt.bar(range(256), hist1, color=color1, alpha=0.5, label=f'{channel_name}')
    mean1 = np.sum(np.arange(256) * hist1) / np.sum(hist1)
    plt.axvline(mean1, color=color1, linestyle='--', label=f'{channel_name}: {mean1:.2f}')
    plt.title(f'Histogram for {channel_name}')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{channel_name}.jpg')
    plt.show()
    plt.close()
    # plt.bar(range(256), hist2, color=color2, alpha=0.5, label='Domain B - Clinic')
    # mean2 = np.sum(np.arange(256) * hist2) / np.sum(hist2)
    # plt.axvline(mean2, color=color2, linestyle='--', label=f'Domain B Mean: {mean2:.2f}')
    # plt.title(f'Histogram for {channel_name} domain B - clinic')
    # plt.xlabel('Intensity Value')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.savefig(f'{channel_name} domain B - clinic.jpg')
    # plt.show()
    # plt.close()
# # Plot histograms for each color channel
# plot_histogram(domain1_r_hist.cpu().data.numpy(), domain2_r_hist.cpu().data.numpy(), 'Red domain A vs B', 'red', 'darkred')
# plot_histogram(domain1_g_hist.cpu().data.numpy(), domain2_g_hist.cpu().data.numpy(), 'Green domain A vs B', 'green', 'darkgreen')
# plot_histogram(domain1_b_hist.cpu().data.numpy(), domain2_b_hist.cpu().data.numpy(), 'Blue domain A vs B', 'blue', 'darkblue')
#
#
# plot_histogram(domain1_r_hist.cpu().data.numpy(), output_to_zero_r_hist.cpu().data.numpy(), 'Red domain A vs B2A', 'red', 'darkred')
# plot_histogram(domain1_g_hist.cpu().data.numpy(), output_to_zero_g_hist.cpu().data.numpy(), 'Green domain A vs B2A', 'green', 'darkgreen')
# plot_histogram(domain1_b_hist.cpu().data.numpy(), output_to_zero_b_hist.cpu().data.numpy(), 'Blue domain  vs B2A', 'blue', 'darkblue')
#
# plot_histogram(domain2_r_hist.cpu().data.numpy(), output_to_one_r_hist.cpu().data.numpy(), 'Red domain B vs A2B', 'red', 'darkred')
# plot_histogram(domain2_g_hist.cpu().data.numpy(), output_to_one_g_hist.cpu().data.numpy(), 'Green domain B vs A2B', 'green', 'darkgreen')
# plot_histogram(domain2_b_hist.cpu().data.numpy(), output_to_one_b_hist.cpu().data.numpy(), 'Blue domain B vs A2B', 'blue', 'darkblue')
#
# plot_histogram(output_to_zero_r_hist.cpu().data.numpy(), output_to_one_r_hist.cpu().data.numpy(), 'Red domain B2A vs A2B', 'red', 'darkred')
# plot_histogram(output_to_zero_g_hist.cpu().data.numpy(), output_to_one_g_hist.cpu().data.numpy(), 'Green domain B2A vs A2B', 'green', 'darkgreen')
# plot_histogram(output_to_zero_b_hist.cpu().data.numpy(), output_to_one_b_hist.cpu().data.numpy(), 'Blue domain B2A vs A2B', 'blue', 'darkblue')

# A-B
plot_histogram(domain1_r_hist.cpu().data.numpy(), 'Red domain A - TCGA', 'red')
plot_histogram(domain1_g_hist.cpu().data.numpy(), 'Green domain A - TCGA', 'green')
plot_histogram(domain1_b_hist.cpu().data.numpy(), 'Blue domain A - TCGA', 'blue')

plot_histogram(domain2_r_hist.cpu().data.numpy(), 'Red domain B - clinic', 'red')
plot_histogram(domain2_g_hist.cpu().data.numpy(), 'Green domain B - clinic', 'green')
plot_histogram(domain2_b_hist.cpu().data.numpy(), 'Blue domain B- clinic', 'blue')

# A' B'
plot_histogram( output_to_zero_r_hist.cpu().data.numpy(), 'Red style transfer Clinic to TCGA',  'darkred')
plot_histogram( output_to_zero_g_hist.cpu().data.numpy(), 'Green style transfer Clinic to TCGA',  'darkgreen')
plot_histogram( output_to_zero_b_hist.cpu().data.numpy(), 'Blue style transfer Clinic to TCGA', 'darkblue')

plot_histogram( output_to_one_g_hist.cpu().data.numpy(), 'Green style transfer TCGA to Clinic', 'darkgreen')
plot_histogram( output_to_one_r_hist.cpu().data.numpy(), 'Red style transfer TCGA to Clinic', 'darkred')
plot_histogram( output_to_one_b_hist.cpu().data.numpy(), 'Blue style transfer TCGA to Clinic', 'darkblue')
