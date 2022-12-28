import argparse
import collections
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm
import model.models as module_arch
from parse_config import ConfigParser
import glob
from utils.fid_score import *
import shutil
import os
import csv
import gc

# def resize(img):
#     return F.interpolate(img, size=256)

# def batch_generate(zs, generator, batch=8):
#     g_images = []
#     with torch.no_grad():
#         for i in range(len(zs)//batch):
#             g_images.append(generator(zs[i*batch:(i+1)*batch]).cpu())
#         if len(zs) % batch > 0:
#             g_images.append(generator(zs[-(len(zs)%batch):]).cpu())
#     return torch.cat(g_images)

# def batch_save(images, folder_name):
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#     for i, image in enumerate(images):
#         vutils.save_image(image.add(1).mul(0.5), f'{folder_name}/{i}.jpg')


def main(config: ConfigParser, args):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    resume = args.resume
    model_name = config['name']
    logger.info(model)
    
    ckpts = glob.glob(os.path.join(resume, "*.pth"))
    # print(glob.glob("checkpoints/*.pth"))
    ckpts.sort()
    os.makedirs(config['eval']['save_dir'], exist_ok=True)
    for i, ckpt in enumerate(ckpts):
        logger.info('Loading checkpoint: {} ...'.format(resume))
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model.DataParallel()
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.generator.eval()

        latent_dim = model.latent_dim

        # generate images
        with torch.no_grad():
            for i in tqdm(range(config['eval']['n_sample']//config['eval']['batch_size'])):
                noise = torch.randn(config['eval']['batch_size'], latent_dim).to(device)
                generated_imgs = model.generator(noise)
                if len(generated_imgs) > 1 and generated_imgs[0].size() != generated_imgs[1].size():
                    generated_imgs = generated_imgs[0]
                for j, g_img in enumerate(generated_imgs):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(config['eval']['save_dir'], '%d.png'%(i*config['eval']['batch_size']+j)))#, normalize=True, range=(-1,1))
                del generated_imgs
                gc.collect()

        fid_value = calculate_fid_given_paths((config['eval']['save_dir'], args.calculated_stats), batch_size=config['eval']['batch_size'], device=device, dims=2048, num_workers=1)

        print(fid_value, ckpt)
        with open(f"./{model_name}-fid.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([fid_value, ckpt.split("/")[-1].split(".")[0]])  

        if args.clear_dir and i != len(ckpts)-1:
            os.remove(ckpt)

    if args.clear_generated:
        shutil.rmtree(config['eval']['save_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating images from trained GAN model')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to checkpoint dir (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-cs', '--calculated_stats', default="None", type=str,
                        help="path to precalculated stats")
    parser.add_argument('-clr', "--clear_dir", default=False, action="store_true",
                        help="whether or not cleaning the whole model's checkpoints dir except for the lastest checkpoint after calculation")        
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    
    options = [
        CustomArgs(['-dir', '--dist'], type=str, target='eval;save_dir'),
        CustomArgs(['-n', '--n_sample'], type=int, target='eval;n_sample'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='eval;batch_size'),
    ]
    config = ConfigParser.from_args(parser)
    # config = ConfigParser.from_args(args)
    # print(type(config.resume), config.resume)
    main(config, args)
