###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os

import torch
import torchvision.transforms as transform

import encoding.utils as utils

from tqdm import tqdm

from torch.utils import data

from encoding.nn import BatchNorm2d
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

def test(args):
    # output folder
    outdir = args.save_folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = get_segmentation_dataset(args.dataset, split=args.split, mode=args.mode,
                                       transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm2d,
                                       base_size = args.base_size, crop_size = args.crop_size, 
                                       batch_size = args.batch_size_per_gpu, up_factor=args.up_factor,
                                       input_channel = args.offset_branch_input_channel, bottleneck_channel=args.bottleneck_channel,
                                       category=args.category, downsampled_input_size=args.downsampled_input_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        if 'val' in args.mode:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image)
                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
