
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from torchvision import transforms
import torch
from metrics.metric import calculate_metrics_folder as C_metrics

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    test_dir = os.path.join(opt.results_dir, '{}_{}_{}'.format(opt.name,opt.phase, opt.epoch))  
    os.makedirs( test_dir, exist_ok=True)
    print('creating test directory to store generation', test_dir)
    to_pil = transforms.ToPILImage()
    print(f' Saving the generation result for step: {opt.num_timesteps}')

    for i, (data,data2) in enumerate(zip(dataset,dataset2)):
        if i == 0:
            model.data_dependent_initialize(data,data2)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data,data2)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results  #return a collections.OrderedDict with key showned in self.visual_names: ('real','fake_1','fake_2','fake_3)
        img_path = model.get_image_paths()[0]  #return is a list
        if i % 500 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        if isinstance(img_path,str):
            image_name = img_path.split('/')[-1]
            save_dir = os.path.join(test_dir,image_name)
            fake_target = visuals[f"fake_1"].squeeze(0)   
            ### modify the pixel value ot (0,1) for pil saving
            fake_target =(fake_target+1)/2
            image = to_pil(fake_target) 
            image.save(save_dir)
        else:
            raise ValueError(f'img_path should be str but got {type(img_path)}')
 
    ### after generate images calculate the psnr and ssim metrics
    print(f'starting calculating the psnr and ssim metrics')
    generated_path = test_dir
    target_path =  opt.target_truth_path
    UNSB_metrics = C_metrics(generated_path,target_path)
    average_ssim, average_psnr = UNSB_metrics.metrics_folder()
    print(f'save metrics value at {os.path.join(opt.metrics_save_dir,opt.metrics_dic_name)}')
    with open(os.path.join(opt.metrics_save_dir,opt.metrics_dic_name),'w') as f:
        opt_dict = vars(opt)
        f.write(f'Result for experiment:{opt.name}'+'\n')
        for k, v in opt_dict.items():
            f.write(f'{k}: {v}'+'\n')
        f.write(f'============'+'\n')
        f.write(f'Average SSIM is:{average_ssim}'+'\n')
        f.write(f'Average PSNR is:{average_psnr}'+'\n')

