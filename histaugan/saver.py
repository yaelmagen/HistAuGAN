import datetime
import logging
import os
import random
import sys

import numpy as np
import torchvision
from PIL import Image
from pytorch_fid import fid_score
from torch.utils.tensorboard import SummaryWriter

import torch
from torchvision import transforms

import os
sys.path.append(f'{os.getcwd()}')
from augmentations import generate_hist_augs

from pytorch_fid import fid_score

logger = logging.getLogger('main_logger')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mapping = {0:1,
           1:0}
folder_mapping = {'trainA':0,
                  'trainB':1}
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
    def __init__(self, opts,val_dataset=0): # todo move val_dataset to set function

        self.check_fid = opts.check_fid
        self.save_interval_track = opts.save_interval_track
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
        self.amount_to_track = opts.amount_to_track
        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.amount_to_track > 0:
            # self.tracking_idx = np.random.choice(range(total_size), size=self.amount_to_track, replace=False)
            self.tracking_idx = random.sample(val_dataset.indices, self.amount_to_track)
            self.tracking_path = os.path.join(self.save_path, 'tracking')
            if not os.path.exists(self.tracking_path):
                os.makedirs(self.tracking_path)
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
                logger.info(f"printing loss: {m}_{mode} , val: {getattr(model, m)} , it: {total_it}")
            # write img
            image_dis = torchvision.utils.make_grid(model.image_display,
                                                    nrow=model.image_display.size(0) // 2) / 2 + 0.5
            self.writer.add_image('Image', image_dis, total_it)

    # save result images
    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:

            logger.info('--- save the images for @ ep %d ---' % (ep))
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
            logger.info('--- save the model @ ep %d ---' % (ep))
            model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
        elif ep == -1:
            model.save('%s/last.pth' % self.model_dir, ep, total_it)

    def load_tensor_image(self,img_path):
        img_from_pangea = Image.open(img_path)
        img_from_pangea = img_from_pangea.convert('RGB')
        convert_tensor = transforms.ToTensor()
        return convert_tensor(img_from_pangea)

    def run_inference(self, ep, model, dataset,idx, train_val_flag):
        self.run_inference_for_same_images( ep, model, dataset,train_val_flag)
        self.run_inference_for_all( ep, model, dataset,idx,train_val_flag)

    def run_inference_for_same_images(self, ep, model, dataset,train_val_flag):
        if train_val_flag == 'val':
            if (ep + 1) % self.save_interval_track == 0 and self.amount_to_track>0:
                logger.info(f'--- run_inference_for_same_images ep {ep}--- ' )
                for domain,all_domain_paths in enumerate(dataset):
                    if domain > 1:
                        break
                    for idx in self.tracking_idx:
                        try:
                            file_path = all_domain_paths[idx]
                            img = self.load_tensor_image(file_path)
                            img = img.to(device)
                            origin_path = str(file_path.split('/')[-1]).replace(".png",
                                                                              f'_original')

                            save_imgs([img], [origin_path], os.path.join(self.tracking_path, str(domain)), True)

                            img = img.unsqueeze(0)
                            out = generate_hist_augs(img, domain, model, z_content=None, same_attribute=False,
                                                     new_domain=mapping[domain],
                                                     stats=None, device=device)
                            new_file_name = str(file_path.split('/')[-1]).replace(".png", f'_d_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_dom_{domain}_ep_{ep}')
                            save_imgs([out[0]], [new_file_name], os.path.join(self.tracking_path, str(domain )),True)

                        except Exception as ex:
                            logger.error(f'error running inference (tracking) for file {file_path} error {ex}')

    def run_inference_for_all(self, ep, model, dataset, idx, train_val_flag):

        if (ep + 1) % self.save_interval == 0:
            logger.info(f'--- running inference for all images ep {ep},{train_val_flag}---')
            new_save_path = os.path.join(self.save_path, train_val_flag)
            if not os.path.exists(new_save_path):
                os.mkdir(new_save_path)

            start = datetime.datetime.now()
            counter = 0
            batch_size = 10
            batch_paths = []
            for domain, all_domain_paths in enumerate(dataset):
                if counter > 30000:
                    break
                all_domain_paths = []
                for i in idx.indices:
                    try:
                        if i < len(dataset[domain]):
                            all_domain_paths.append(dataset[domain][i])
                    except Exception as ex:
                        logger.warning(f'error running inference (all)  error {ex} for id {i}')
                lap = start

                for file_path in all_domain_paths:
                    try:
                        batch_paths.append(file_path)
                        if len(batch_paths) == batch_size:
                            # run infrence
                            # Load batch of images
                            imgs = torch.stack([self.load_tensor_image(p) for p in batch_paths])
                            imgs = imgs.to(device)
                            # img = self.load_tensor_image(file_path)
                            # img = img.to(device)
                            if domain>1:
                                break
                            new_domain = mapping[domain]
                            outs = generate_hist_augs(imgs, domain, model, z_content=None, same_attribute=False,
                                                      new_domain=new_domain,
                                                      stats=None, device=device)
                        else:
                            continue
                        new_file_names = []
                        # save
                        for _, file_path in zip(outs, batch_paths):
                            if self.overwrite_save:
                                new_file_name = str(file_path.split('/')[-1]).replace(".png",
                                                                                      f'from_dom_{domain}_to_dom_{new_domain}')
                            else:
                                new_file_name = str(file_path.split('/')[-1]).replace(".png",
                                                                                      f'_from_dom_{domain}_to_dom_{new_domain}_ep_{ep}')
                            new_file_names.append(new_file_name)
                        new_file_path = os.path.join( new_save_path, str(domain))
                        save_imgs(outs, new_file_names, new_file_path, True)

                        counter += len(batch_paths)
                        batch_paths = []
                        if counter % 5000 == 0:
                            elapsed_time = datetime.datetime.now() - lap
                            logger.info(f'saved 5000 images in {(elapsed_time.total_seconds())} seconds ---')
                            lap = datetime.datetime.now()
                    except Exception as ex:
                        logger.error(f'error running inference (all) for file {file_path} error {ex}')
                counter = 0
            elapsed_time = datetime.datetime.now() - start
            logger.info(f'saved total of {counter} images in {(elapsed_time.total_seconds())} seconds ---')

    def run_fid(self,ep,train_val_flag):
        if (ep + 1) % self.save_interval == 0 and self.check_fid:
            logger.info(f'--- running fid ep {ep} , {train_val_flag}---')
            for domain,folder in enumerate( os.listdir(self.train_images_path)):
                try:
                    if folder.__contains__('A') or folder.__contains__('B'):
                        real_images = os.path.join(self.train_images_path,folder)
                        if folder in folder_mapping:
                            generated_images =  os.path.join(os.path.join(self.save_path,train_val_flag), str(mapping[domain]))
                        fid_value = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=50, device=device,
                                                                dims=2048)

                        self.writer.add_scalar(f"fid_r_{folder}_d_{mapping[domain]}_{train_val_flag}", fid_value, ep)
                        logger.info(f"fid folder {folder} to {generated_images} , score {fid_value}")
                        logger.info(f"---"*20)

                except Exception as ex:
                    logger.error(f'error running fid for folders  {real_images} and {generated_images} error {ex}')
