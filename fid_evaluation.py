
from pytorch_fid import fid_score
import argparse

def fid_evaluation(real_images, generated_images):
    # add these parameters to the run:
    # --real_images C:/Users/Admin/PycharmProjects/Pangea/HistAuGAN/data/trainA
    # --generated_images C:/Users/Admin/PycharmProjects/Pangea/HistAuGAN/histaugan/results/generated_images

    # FID score between the two sets of images as numpy arrays: real_images and generated_images
    fid_value = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=1, device=0, dims=64, num_workers=0)
    print("FID Score:", fid_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_images', metavar='path', help='path to real images', required=True)
    parser.add_argument('--generated_images', metavar='path', help='path to generated images', required=True)
    args=parser.parse_args()
    fid_evaluation(real_images=args.real_images, generated_images=args.generated_images)

