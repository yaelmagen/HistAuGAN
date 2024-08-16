import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # Import tqdm
import plotly.graph_objs as go
from bokeh.plotting import figure, show, output_file,save
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

transform = transforms.Compose([
    transforms.ToTensor()
])

# Paths to folders and pickle file
folder1_path = '/data/data/proc-images/trainA'
folder2_path = '/data/data/proc-images/trainB'
pickle_path = 'features_labels_all.pkl'  # Path to the pickle file

# Function to extract features
def extract_features(dataloader):
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            output = model(batch).squeeze()
            features.append(output.cpu().numpy())
    return np.concatenate(features)

# Check if pickle file exists
if os.path.exists(pickle_path):
    print("Loading features and labels from pickle file...")
    with open(pickle_path, 'rb') as f:
        all_features, labels = pickle.load(f)
else:
    # Load datasets
    fraction = 1
    dataset1 = ImageFolderDataset(folder1_path, transform)
    dataset2 = ImageFolderDataset(folder2_path, transform)
    subset_length1 = int(len(dataset1) * fraction)
    subset_length2 = int(len(dataset2) * fraction)

    subset1 = torch.utils.data.Subset(dataset1, list(range(subset_length1)))
    subset2 = torch.utils.data.Subset(dataset2, list(range(subset_length2)))

    # Create DataLoaders
    batch_size = 24  # Adjust this based on your GPU memory
    dataloader1 = DataLoader(subset1, batch_size=batch_size, shuffle=False)
    dataloader2 = DataLoader(subset2, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    # Extract features
    print("Extracting features from dataset 1...")
    features1 = extract_features(dataloader1)

    print("Extracting features from dataset 2...")
    features2 = extract_features(dataloader2)

    # Combine features and create labels
    all_features = np.concatenate((features1, features2))
    labels = np.concatenate((np.zeros(len(features1)), np.ones(len(features2))))

    # Save features and labels to a pickle file
    with open(pickle_path, 'wb') as f:
        pickle.dump((all_features, labels), f)
    print("Features and labels saved to pickle file.")

# Initialize UMAP for 3D
umap_model = UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42
)

# Fit and transform data with UMAP
print("Running UMAP...")
umap_results = umap_model.fit_transform(all_features)

trace = go.Scatter3d(
    x=umap_results[:, 0],
    y=umap_results[:, 1],
    z=umap_results[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=labels,  # Color by labels
        colorscale='Viridis',
        opacity=0.8
    )
)

layout = go.Layout(
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig = go.Figure(data=[trace], layout=layout)

# Save the figure as an HTML file
plotly_output_path = "umap_plotly_new.html"
fig.write_html(plotly_output_path)
print(f"Plotly figure saved to {plotly_output_path}")