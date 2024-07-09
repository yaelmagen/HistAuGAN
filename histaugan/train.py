# for local training
# add a folder 'data' with folders of domains trainA trainB trainC ....
# add these parameters to the run --dataroot C:\ydata\pangea\HistAuGAN\data  --name train.log --num_domains 2 --batch_size 2 --nThreads 1 --n_ep 10
import logging
import time

import torch

from datasets import dataset_multi
from model import MD_multi
from options import TrainOptions
from saver import Saver
from torch.utils.data import DataLoader, random_split


def main():
    logger = setup_logger()
    start = time.time()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.autograd.set_detect_anomaly(True)
    change_domain_num = True
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # data loader
    # if True:
    #     tmp_dir = Path('/localscratch/sophia.wagner') / Path(opts.dataroot).name
    #     if not tmp_dir.is_dir():
    #         start = time.time()
    #         print(f'--- copying all images to {tmp_dir}  ------------')
    #         tmp_dir.mkdir(parents=True)
    #         for file in Path(opts.dataroot).glob('*.txt'):
    #             sub_dir = tmp_dir / file.stem
    #             sub_dir.mkdir()
    #             for img_path in file.read_text().split('\n')[:-1]:
    #                 target_path = sub_dir / Path(img_path).name
    #                 shutil.copyfile(img_path, target_path)
    #         print(f'--- copied all image to {tmp_dir} in {int(time.time() - start)}s ------------')
    #     opts.dataroot = str(tmp_dir)
    logger.info('\n--- load dataset ---')
    # dataset = dataset_multi_from_txt(opts)
    dataset = dataset_multi(opts)
    total_size = len(dataset)
    train_size = int((1 - opts.val_split) * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads)
    dataset_mapping = {'train':train_dataset, 'val':val_dataset}
    logger.info(f'------ took {int(time.time() - start)}s until here')

    # model
    logger.info('\n--- load model ---')
    model = MD_multi(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume,True,change_domain_num)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    logger.info('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts,val_dataset)

    logger.info(f'------ took {int(time.time() - start)}s until here')

    # train
    logger.info('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        logger.info(f'start train ep {ep}')
        model.train()  # Set the model to training mode
        for it, (images, c_org) in enumerate(train_loader):
            if images.size(0) != opts.batch_size:
                continue

            # input data
            images = images.cuda(opts.gpu).detach()
            c_org = c_org.cuda(opts.gpu).detach()
            # c_trg = c_trg.cuda(opts.gpu).detach()
            # input()

            # update model
            if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                model.update_D_content(images, c_org)
                continue
            else:
                model.update_D(images, c_org)
                model.update_EG()

            # save to display file
            if not opts.no_display_img:
                saver.write_display(ep, model)

            logger.info('total_it: %d (ep %d, it %d), lr %08f' %
                  (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            # if total_it >= max_it:
            #     saver.write_img(-1, model)
            #     saver.write_model(-1, total_it, model)
            #     break
        logger.info(f'finish train ep {ep}')
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            logger.info(f'start validation  ep {ep}')
            for it, (images, c_org) in enumerate(val_loader):
                if images.size(0) != opts.batch_size:
                    continue

                # input data
                images = images.cuda(opts.gpu).detach()
                c_org = c_org.cuda(opts.gpu).detach()

                if (it + 1) % opts.d_iter != 0 and it < len(val_loader) - 2:
                    model.update_D_content(images, c_org, isVal=True)
                    continue
                else:
                    model.update_D(images, c_org, isVal=True)
                    model.update_EG(isVal=True)

                # save to display file
                if not opts.no_display_img:
                    saver.write_display(ep, model, mode='val')
        logger.info(f'finish validation  ep {ep}')
        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        saver.write_model(ep, total_it, model)
        for train_val_flag in dataset_mapping.keys():
            saver.run_inference(ep, model, dataset.images,dataset_mapping[train_val_flag],train_val_flag)
            saver.run_fid(ep,train_val_flag)

    return


def setup_logger(log_file='output.log'):
    # Create a logger
    logger = logging.getLogger('main_logger')

    # Only add handlers if they haven't been added before
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.propagate = False
    return logger
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
