import os

import numpy as np
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import torch
from torchvision import transforms
from augmentations import generate_hist_augs

from pytorch_fid import fid_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# tensor to PIL Image
def tensor2img(img,partial=False):
    if partial:
        img = img.cpu().float().numpy()
    else:
        img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


# save a set of images
def save_imgs(imgs, names, path,partial=False):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img,partial)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))


def save_concat_imgs(imgs, name, path):
    if not os.path.exists(path):
        os.mkdir(path)
    imgs = [tensor2img(i) for i in imgs]
    widths, heights, c = zip(*(i.shape for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in imgs:
        im = Image.fromarray(im)
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(os.path.join(path, name + '.png'))


class Saver():
    def __init__(self, opts):

        self.display_dir = os.path.join(opts.display_dir, opts.name)
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq
        self.save_path = opts.save_path
        self.save_interval = opts.save_interval
        self.train_images_path = opts.dataroot
        self.overwrite_save = opts.overwrite_save
        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model ,mode= "train"):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if
                       not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                # print(f'm: {m},getattr(model, m): {getattr(model, m)} iter: {total_it}')
                self.writer.add_scalar(f"{m}_{mode}", getattr(model, m), total_it)
            # write img
            image_dis = torchvision.utils.make_grid(model.image_display,
                                                    nrow=model.image_display.size(0) // 2) / 2 + 0.5
            self.writer.add_image('Image', image_dis, total_it)

    # save result images
    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
        elif ep == -1:
            model.save('%s/last.pth' % self.model_dir, ep, total_it)

    def load_tensor_image(self,img_path):
        img_from_pangea = Image.open(img_path)
        img_from_pangea = img_from_pangea.convert('RGB')
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img_from_pangea)
    def run_infrence(self,ep,model,dataset):
        #todo add save infernece for X pictures for every other batch to see progress save timestamp to see progress
        if (ep + 1) % self.save_interval == 0:
            for domain,all_domain_paths in enumerate(dataset):
                for file_path in all_domain_paths:

                    #run infrence
                    img = self.load_tensor_image(file_path)
                    img = img.to(device)
                    new_domain = 1 if domain == 0 else 0
                    out = generate_hist_augs(img, domain, model, z_content=None, same_attribute=False, new_domain=new_domain,
                                             stats=None, device=device)
                    # save
                    if self.overwrite_save:
                        new_file_name = str(file_path.split('/')[-1]).replace(".png", f'_dom_{domain + 1}.png')
                    else:
                        new_file_name = str(file_path.split('/')[-1]).replace(".png", f'_dom_{domain + 1}_ep_{ep}.png')
                    save_imgs([out], [new_file_name], os.path.join(self.save_path, str(domain+1)))

                    # Compute FID score
                    fid_value = fid_score.calculate_fid_given_paths(file_path, os.path.join(self.save_path, str(domain+1)))
                    print(f'FID: {fid_value}')




