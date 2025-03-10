import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional
from tqdm import tqdm
from transformers import AutoImageProcessor
from data.dataset import build_dataloader
from model.encoder import get_lora_model
from model.segmentation_head import SegmentationHead
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import wandb
import argparse

from util import calculate_iou, calculate_objective, fix_randseed, log_info, print_trainable_parameters, to_cuda, calculate_f1

BASE_MODEL_DICT = {
    'dino' : 'facebook/dinov2-large',
    'radio' : 'nvidia/RADIO-H',
    'sam' : 'facebook/sam-vit-large',
    'apple' : 'apple/aimv2-large-patch14-224'
}


def run_epoch(model, dataloader, optimizer, batch_size=16):
        # Function to handle training and evaluation
        total_loss = total_iou = total_F1 = 0

        is_train = optimizer is not None

        # Set model mode
        if is_train:
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluation mode

        # Loop over the current dataloader
        nsamples = 0

        for batch in tqdm(dataloader):
            batch = to_cuda(batch)
            img = batch['img']
            img = torchvision.transforms.functional.resize(img, [s*3 for s in img.shape[-2:]],torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
            mask = batch['mask']

            nsamples += img.shape[0]

            if is_train:
                optimizer.zero_grad()  # Zero out the gradients during training
                pred = model(img)
                loss = calculate_objective(pred, mask)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():  # No gradient calculation during validation/testing
                    pred = model(img)
                    loss = calculate_objective(pred, mask)

            total_loss += loss.item()
            total_iou += calculate_iou(pred, mask)
            total_F1 += calculate_f1(pred, mask)[0]
        
        return total_loss/nsamples, total_iou/nsamples, total_F1/nsamples

def experiment(args):
    wandb.init(entity='rafael-sterzinger', project='few_shot_map_seg', id=args.exp_name if args.exp_name else None)

    base_model_path = BASE_MODEL_DICT[args.base_model]
    base_model = get_lora_model(base_model_path)
    model = SegmentationHead(base_model, args.base_model).cuda()
    print_trainable_parameters(model)

    processor = AutoImageProcessor.from_pretrained(base_model_path)
    dl_train, dl_val, dl_test = build_dataloader(args, processor, args.mixup, args.cutmix)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        train_loss, train_iou, train_f1 = run_epoch(model, dl_train, optimizer)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_miou": train_iou, "train_f1": train_f1})
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}, mIoU: {train_iou:.4f}, F1: {train_f1:.4f}")
        val_loss, val_iou, val_f1 = run_epoch(model, dl_val, None)
        wandb.log({"epoch": epoch + 1, "val_loss": val_iou, "val_miou": val_iou, "val_f1": val_f1})
        print(f"VALIDATION: Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}, F1: {val_f1:.4f}")

    test_loss, test_iou, test_f1 = run_epoch(model, dl_test, None, epoch)
    wandb.log({"epoch": epoch+1, "loss": test_loss, "miou": test_iou, "f1": test_f1})
    print(f"TESTING: Loss: {test_loss:.4f}, mIoU: {test_iou:.4f}, F1: {test_f1:.4f}")
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, choices=['railway', 'vineyard'], help="Chose railways or vineyards")
    parser.add_argument("--base_model", type=str, choices=['dino', 'sam', 'radio', 'apple'])
    parser.add_argument("--decoder", type=str, choices=[''])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--nshots", default=10)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    fix_randseed(args.seed)
    log_info(args)

    wandb.disabled = not args.exp_name
    
    experiment(args)
