import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
from data.unpaired_dataset import ValidationSet
from torch.utils.data import Dataset, DataLoader
from metrics.metric import Normalization, calculate_ssim
from torchmetrics.functional import structural_similarity_index_measure as ssim


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1
    ###
    if opt.if_validation ==True: # track validation result during training
        print(f'Creating Validation Dataloder based on set:{opt.validation_dict_path}')
        vali_data = ValidationSet(opt)
        validation_dataloader = DataLoader(vali_data,batch_size=opt.validation_batch,shuffle=False,drop_last=False) ##validation
        print(f'Validation set has been created')
    ###
    times = []
    best_ssim = opt.best_SSIM ## best ssim value for validation, need to modify when continue training  ###
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, (data,data2) in enumerate(zip(dataset,dataset2)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size  ### how many images has been processed ###
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0: ### intialize the network start from opt.epoch_count ###
                model.data_dependent_initialize(data,data2)  
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data,data2)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        ### calculate the psnr,ssim over validation set att the end of every epoch
        if opt.if_validation ==True and opt.validation_phase == True:
            model.eval() ## set to eval state
            ssim_value = 0 ## ssim value for each validation epoch ###
            opt.phase = 'test' # modify the phase for validation
            with torch.no_grad():
                for i,data in enumerate(validation_dataloader): #data is a dict contain 2 tensor(b,c,h,w) and 2 image list
                    model.set_input(data)
                    model.forward() ## call forward function to get related interval value
                    fake_target = getattr(model,f'fake_1')
                    true_target = model.real_B
                    ssim_value += ssim(fake_target,true_target,data_range=(-1.0,1.0))

            ssim_best = ssim_value / len(validation_dataloader)
            print(f'Epochs:{epoch}: average validation SSIM:{ssim_best}')
            if ssim_best > best_ssim:
                best_ssim = ssim_best
                model.save_networks(f'ssim_best')
                print(f'Saving best Epochs:{epoch}: best SSIM:{best_ssim}')

            model.train() # set back to training  
            opt.phase = 'train' # set phase back to training   


