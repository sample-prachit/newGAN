import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment

#Vajira
from custom_data import ImageAndMaskDataFromSinGAN

policy = 'color,translation'
import lpips

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Move device initialization to top
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Initialize LPIPS on the correct device once
percept = lpips.LPIPS(net='vgg').to(device)


#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real", scaler=None):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        data = data.contiguous()
        # Update deprecated autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
            
            # Handle single channel data
            data_img = data[:,0:1, :, :].contiguous()
            rec_all_img = rec_all[:, 0:1, :, :].contiguous()
            rec_small_img = rec_small[:, 0:1, :, :].contiguous()
            rec_part_img = rec_part[:, 0:1, :, :].contiguous()

            interpolated_data = F.interpolate(data_img, rec_all.shape[2]).contiguous().to(device)
            interpolated_small = F.interpolate(data_img, rec_small.shape[2]).contiguous().to(device)
            interpolated_part = F.interpolate(crop_image_by_part(data_img, part), rec_part.shape[2]).contiguous().to(device)

            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
                percept(rec_all_img.to(device), interpolated_data).sum() + \
                percept(rec_small_img.to(device), interpolated_small).sum() + \
                percept(rec_part_img.to(device), interpolated_part).sum()
        
        scaler.scale(err).backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            data = [d.contiguous() if isinstance(d, torch.Tensor) else [t.contiguous() for t in d] for d in data]
            pred = net(data, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        scaler.scale(err).backward()
        return pred.mean().item()
        

def train(args):
    # Network configuration
    nz = 256  # latent vector size
    ngf = 64  # generator feature size
    ndf = 64  # discriminator feature size
    
    # Optimize hyperparameters for 20 images
    total_iterations = 10000  # Reduce iterations
    batch_size = 10  # Smaller batch size for better stability
    im_size = args.im_size
    dataloader_workers = 2  # Reduce workers
    
    # Learning rate schedule
    base_lr = 0.0001  # Lower learning rate for stability
    nbeta1 = 0.5
    nbeta2 = 0.999
    
    # Gradient clipping value
    grad_clip = 0.5
    
    # More frequent validation to catch instability
    save_interval = 200
    validation_interval = 500
    
    # Enable TF32 for better performance/stability balance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    checkpoint = args.ckpt
    im_size = args.im_size
    use_cuda = False
    multi_gpu = False  # Set to False since we're using CPU
    current_iteration = 0
    
    # Add early stopping
    best_fid = float('inf')
    patience = 5
    no_improve = 0
    
    saved_model_folder, saved_image_folder = get_dir(args)
    
    # Update device initialization to properly check for MPS
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(), # removed by Vajira, check the dataloader
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # removed by Vajira, check the dataloader
        ]
    trans = transforms.Compose(transform_list)
    
    #if 'lmdb' in data_root:
    #    from operation import MultiResolutionDataset
    #    dataset = MultiResolutionDataset(data_root, trans, 1024)
    #else:
        #dataset = ImageFolder(root=data_root, transform=trans)
    try:
        dataset = ImageAndMaskDataFromSinGAN(args.path_img, args.path_mask, transform=trans)
        print(f"Initial dataset size: {len(dataset)}")
        
        if args.num_imgs_to_train == -1:
            indices = range(len(dataset))
        else:
            if args.num_imgs_to_train > len(dataset):
                raise ValueError(f"Requested {args.num_imgs_to_train} images but only {len(dataset)} available")
            indices = range(args.num_imgs_to_train)
        
        dataset = Subset(dataset, indices)
        print(f"Using {len(dataset)} images for training")
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
            
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        raise

    # Reduce batch size if needed
    effective_batch_size = min(args.batch_size, len(dataset))
    if effective_batch_size != args.batch_size:
        print(f"Warning: Reducing batch size to {effective_batch_size} to match dataset size")
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Pin memory and enable async data loading with optimal settings
    dataloader = iter(DataLoader(dataset, 
                               batch_size=effective_batch_size,
                               shuffle=False,
                               sampler=InfiniteSamplerWrapper(dataset),
                               num_workers=dataloader_workers,
                               pin_memory=True,
                               persistent_workers=True,
                               prefetch_factor=4))
    
    # Enable automatic mixed precision for faster training
    scaler = torch.amp.GradScaler()
    
    # Pre-fetch data to GPU
    next_data = next(dataloader).to(device)
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=args.nc)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, nc=args.nc)
    netD.apply(weights_init) 

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    # Update optimizers with better parameters
    optimizerG = optim.AdamW(
        netG.parameters(),
        lr=base_lr,
        betas=(nbeta1, nbeta2),
        weight_decay=0.01
    )
    optimizerD = optim.AdamW(
        netD.parameters(),
        lr=base_lr,
        betas=(nbeta1, nbeta2),
        weight_decay=0.01
    )
    
    # Add learning rate schedulers
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, total_iterations)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, total_iterations)
    
    backup_para = None  # Initialize backup_para
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next_data
        # Pre-fetch next batch asynchronously
        next_data = next(dataloader).to(device, non_blocking=True)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
        
        # Use AMP for forward/backward passes
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            fake_images = netG(noise)
            if isinstance(fake_images, list):
                fake_images = [f.contiguous() for f in fake_images]

            real_image = DiffAugment(real_image, policy=policy)
            fake_images = [DiffAugment(fake.contiguous(), policy=policy) for fake in fake_images]
            
            ## 2. train Discriminator
            netD.zero_grad()

            err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real", scaler=scaler)
            train_d(netD, [fi.detach() for fi in fake_images], label="fake", scaler=scaler)
            
            # Update discriminator
            scaler.unscale_(optimizerD)
            scaler.step(optimizerD)
            
            ## 3. train Generator
            netG.zero_grad()
            pred_g = netD(fake_images, "fake")
            err_g = -pred_g.mean()

            # Scale and backprop generator loss
            scaler.scale(err_g).backward()
            scaler.unscale_(optimizerG)
            scaler.step(optimizerG)
            scaler.update()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(netD.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), grad_clip)
            
            # Update learning rates
            schedulerG.step()
            schedulerD.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        # Monitor GPU memory usage
        if iteration % 100 == 0:
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"Current LR: {schedulerG.get_last_lr()[0]:.6f}")
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % validation_interval == 0:
            # Monitor losses
            if err_dr < 0.1 or err_g > 10:
                print("Warning: Training may be unstable")
            
            # Save latest images
            if iteration % (save_interval*10) == 0:
                backup_para = copy_G_params(netG)  # Create backup before saving
                load_params(netG, avg_param_G)
                
                with torch.no_grad():
                    # Save generated samples - both channels
                    gen_output = netG(fixed_noise)[0]
                    vutils.save_image(gen_output[:,0:1].add(1).mul(0.5), 
                                    saved_image_folder+'/%d_img.png'%iteration, nrow=4)
                    vutils.save_image(gen_output[:,1:2].add(1).mul(0.5), 
                                    saved_image_folder+'/%d_mask.png'%iteration, nrow=4)
                    
                    # Save reconstruction samples
                    vutils.save_image(torch.cat([
                            F.interpolate(real_image[:,0:1], 128),  # Image channel 
                            rec_img_all[:,0:1],  # Reconstructed image
                            rec_img_small[:,0:1],
                            rec_img_part[:,0:1]]).add(1).mul(0.5), 
                            saved_image_folder+'/rec_img_%d.png'%iteration)
                    
                    # Save mask reconstructions
                    vutils.save_image(torch.cat([
                            F.interpolate(real_image[:,1:2], 128),  # Mask channel
                            rec_img_all[:,1:2],  # Reconstructed mask
                            rec_img_small[:,1:2],
                            rec_img_part[:,1:2]]).add(1).mul(0.5), 
                            saved_image_folder+'/rec_mask_%d.png'%iteration)
                
                if backup_para is not None:  # Only restore if backup exists
                    load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=1, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test_4ch_num_img_5', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=10, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    # new parameters- added to process 4 channels data
    parser.add_argument("--nc", type=int, default=2, help="number of channels in input images (1 for image, 1 for mask)")
    parser.add_argument("--path_img", default="./custom_dataset/images", help="image directory")
    parser.add_argument("--path_mask", default="./custom_dataset/masks", help = "mask directory")
    parser.add_argument("--num_imgs_to_train", default=5, type=int, help="number of samples to train. -1 for use all")


    args = parser.parse_args()
    print(args)

    train(args)
