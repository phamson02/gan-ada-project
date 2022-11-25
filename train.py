import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.models as module_arch
import augment as module_augment
import trainer as module_trainer
from parse_config import ConfigParser
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 90
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model.to(device)
    if len(device_ids) > 1:
        model.DataParallel(device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params_G = filter(lambda p: p.requires_grad, model.generator.parameters())
    trainable_params_D = filter(lambda p: p.requires_grad, model.discriminator.parameters())
    optimizer_G = config.init_obj('optimizer_G', torch.optim, trainable_params_G)
    optimizer_D = config.init_obj('optimizer_D', torch.optim, trainable_params_D)
    lr_scheduler_G = config.init_obj('lr_scheduler_G', torch.optim.lr_scheduler, optimizer_G)
    lr_scheduler_D = config.init_obj('lr_scheduler_D', torch.optim.lr_scheduler, optimizer_D)

    # choose augment options
    augment = config.init_obj('augment', module_augment) if bool(config['augment']) else None

    trainer = getattr(module_trainer, config['trainer']['type'])(model, criterion, metrics, optimizer_G, optimizer_D,
                                                                 config=config,
                                                                 device=device,
                                                                 data_loader=data_loader,
                                                                 augment=augment,
                                                                 lr_scheduler_G=lr_scheduler_G,
                                                                 lr_scheduler_D=lr_scheduler_D)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='GAN Trainer')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
