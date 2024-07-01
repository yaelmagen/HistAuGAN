# for local training
# add a folder 'data' with folders of domains trainA trainB trainC ....
# add these parameters to the run
# #for vm: --dataroot /data/dataset  --name /data/train_results/yael.log
# #--name train.log
# --num_domains 2 --batch_size 2 --nThreads 1 --n_ep 10
import time

import torch

from datasets import dataset_multi
from model import MD_multi
from options import TrainOptions
from saver import Saver
from torch.utils.data import DataLoader, random_split


def main():
    start = time.time()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.autograd.set_detect_anomaly(True)

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
    print('\n--- load dataset ---')
    # dataset = dataset_multi_from_txt(opts)
    dataset = dataset_multi(opts)
    total_size = len(dataset)
    train_size = int((1 - opts.val_split) * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.nThreads)

    print(f'------ took {int(time.time() - start)}s until here')

    # model
    print('\n--- load model ---')
    model = MD_multi(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)

    print(f'------ took {int(time.time() - start)}s until here')

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
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

            print('total_it: %d (ep %d, it %d), lr %08f' %
                  (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, total_it, model)
                break

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
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

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        saver.write_model(ep, total_it, model)
        saver.run_infrence(ep, model, dataset.images)
    return


if __name__ == '__main__':
    print('hi')
    main()
