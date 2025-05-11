import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import copy
import random
import string
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms.functional
from segmentation_models_pytorch import Unet
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor
from data.dataset import build_dataloader
from model.encoder import get_lora_model
from model.decoder import SegmentationHead
from patchify import unpatchify
import wandb
import argparse

from util import calculate_iou, calculate_objective, fix_randseed, log_info, print_trainable_parameters, to_cuda, calculate_f1, EarlyStopping

BASE_MODEL_DICT = {
    'dino' : 'facebook/dinov2-large',
    'radio_l' : 'nvidia/RADIO-L',
    'radio_h' : 'nvidia/RADIO-H',
    'sam' : 'facebook/sam-vit-large',
    'apple' : 'apple/aimv2-large-patch14-224'
}



def run_epoch(model, dataloader, optimizer, scheduler, scale_factor = 1, save_results = False):
        # Function to handle training and evaluation
        total_loss = total_iou = total_F1 = 0

        is_train = optimizer is not None

        # Set model mode
        if is_train:
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluation mode

        f = None
        if type(model) is Unet:
                f = model
                model = lambda x: f(x).squeeze(1)


        # Loop over the current dataloader
        nsamples = 0

        if args.class_name == "icdar":
            preds = []

        for batch in tqdm(dataloader):
            batch = to_cuda(batch)
            img = batch['img']
            if not f and model.enc_name != "sam" and model.enc_name != "apple":
                img = torchvision.transforms.functional.resize(img, [s*scale_factor for s in img.shape[-2:]],torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
            mask = batch['mask']

            nsamples += img.shape[0]

            if is_train:
                optimizer.zero_grad()  # Zero out the gradients during training
                pred = model(img)
                loss = calculate_objective(pred, mask)
                loss.backward()

                if type(f) is Unet:
                    torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)  
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  

                optimizer.step()
                scheduler.step()
            else:
                with torch.no_grad():  # No gradient calculation during validation/testing
                    #pca = model.visualize_pca(img)
                    pred = model(img)
                    if save_results:
                        for i in range(img.shape[0]):
                            torchvision.utils.save_image((pred[i]>=0.5).float(), f'out/{batch["name"][i]}_{args.nshots}_pred.png')
                    if args.class_name == "icdar":
                        temp_pred = torchvision.transforms.functional.resize(pred, [s*2 for s in pred.shape[-2:]],torchvision.transforms.InterpolationMode.BILINEAR)
                        preds.append(temp_pred.cpu().numpy())
                    loss = calculate_objective(pred, mask)

            total_loss += loss.item()
            total_iou += calculate_iou(pred, mask)
            total_F1 += calculate_f1(pred, mask)[0]
        
        if args.class_name == "icdar" and (dataloader.dataset.split == "val" or dataloader.dataset.split == "test"):
            benchmark = 'icdar'
            import numpy as np
            preds = np.concatenate(preds, axis=0)
            images = []
            start_idx = dataloader.dataset.start_indices
            for i in range(len(start_idx)-1):
                images.append(preds[start_idx[i]:start_idx[i+1]])
            for i, img in enumerate(images):
                name = dataloader.dataset.images[i]['name']
                shape = dataloader.dataset.images[i]['patch_shape']
                orig_shape = dataloader.dataset.images[i]['orig_size']
                frame_mask = dataloader.dataset.images[i]['frame_mask']
                PH, PW, H, W, C = shape  # Number of patches in height & width, patch size, and channels
                step = H // 4  # Overlapping step size (448//2 = 224)

                # Initialize empty arrays for the recombined image and weight matrix
                full_image = np.zeros((PH * step + step, PW * step + step))
                weight_matrix = np.zeros((PH * step + step, PW * step + step))

                # Create a Gaussian weight map for smooth blending
                y_coords, x_coords = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
                gaussian_weights = np.exp(-4 * (x_coords**2 + y_coords**2))

                # Reshape patches back into their spatial arrangement
                image_patches = img.reshape(PH, PW, H, W)

                # Reconstruct the full image by placing patches with Gaussian blending
                for ph in range(PH):
                    for pw in range(PW):
                        y, x = ph * step, pw * step
                        full_image[y:y+H, x:x+W] += image_patches[ph, pw] * gaussian_weights
                        weight_matrix[y:y+H, x:x+W] += gaussian_weights  # Smarter weighting

                # Normalize overlapping regions by weighted averaging
                full_image /= np.maximum(weight_matrix, 1e-8)
                image_with_frame = full_image[0:orig_shape[0],0:orig_shape[1]]
                image = image_with_frame * (frame_mask/255)
                if not os.path.exists(f'out/{benchmark}'):
                    os.makedirs(f'out/{benchmark}')
                cv2.imwrite(f'out/{benchmark}/{name}-OUTPUT-PRED.png', ((image>0.5)*255).astype(np.uint8))
                cv2.imwrite(f'out/{benchmark}/{name}_soft.png', (image*255).astype(np.uint8))
                cv2.imwrite(f'out/{benchmark}/{name}.png', (image*255).astype(np.uint8))
        
        return total_loss/nsamples, total_iou/nsamples, total_F1/nsamples

def experiment(args):
    if 'unet' not in args.base_model:
        base_model_path = BASE_MODEL_DICT[args.base_model]
        base_model = get_lora_model(base_model_path, args.adapter, type(args.nshots) is int)
        model = SegmentationHead(base_model, args.base_model).cuda()
        preprocessor = AutoImageProcessor.from_pretrained(base_model_path)
    else:
        model = Unet(encoder_name="resnet50", encoder_weights="imagenet", classes=1, activation="sigmoid").cuda()
        preprocessor = transforms.Compose([
        transforms.ToTensor(),                  
        transforms.Normalize(
             mean=[0.485, 0.456, 0.406],         
             std=[0.229, 0.224, 0.225]          
         )
        ])
    param_infos = print_trainable_parameters(model)
    wandb.config.update(param_infos)

    dl_train, dl_val, dl_test = build_dataloader(args, preprocessor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr/10, betas=(0.9, 0.999), weight_decay=0.01)

    # One-Cycle LR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(dl_train))

    best_model = None
    best_iou = 0

    for epoch in range(args.epochs):
        train_loss, train_iou, train_f1 = run_epoch(model, dl_train, optimizer, scheduler, args.scale_factor)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_miou": train_iou, "train_f1": train_f1})
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}, mIoU: {train_iou:.4f}, F1: {train_f1:.4f}")
        if epoch % 10 == 0 or args.nshots != 10:
            val_loss, val_iou, val_f1 = run_epoch(model, dl_val, None, None, args.scale_factor)
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_miou": val_iou, "val_f1": val_f1})
            print(f"VALIDATION: Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}, F1: {val_f1:.4f}")


        if best_model is None or best_iou < val_iou:
            best_iou = val_iou
            best_model = copy.deepcopy(model.state_dict())
    
    torch.save(best_model, f'checkpoint/{args.class_name}_{args.base_model}_{args.adapter}_{args.nshots}_{args.exp_name}.pth')
    model.load_state_dict(best_model)

    test_loss, test_iou, test_f1 = run_epoch(model, dl_test, None, None, args.scale_factor, args.save_results)
    wandb.log({"test_loss": test_loss, "test_miou": test_iou, "test_f1": test_f1})
    print(f"TESTING: Loss: {test_loss:.4f}, mIoU: {test_iou:.4f}, F1: {test_f1:.4f}")

    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, choices=['railway', 'vineyard', 'icdar'])
    parser.add_argument("--adapter", type=str, default='none', choices=['lora', 'lokr', 'loha', 'dora', 'none'], help="Low-rank adaptation methods")
    parser.add_argument("--base_model", type=str, choices=['dino', 'sam', 'radio_l', 'radio_h', 'apple', 'unet'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--nshots", default=1.0)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument("--scale_factor", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    if type(args.nshots) is str:
        if '.' in args.nshots:
            args.nshots = float(args.nshots)
        else:
            args.nshots = int(args.nshots)

    os.environ['WANDB_SILENT']="true" if not args.exp_name else "false"
    random_string = ''.join(random.choices(string.ascii_lowercase, k=4))
    id = f"{args.exp_name}_{random_string}"
    wandb.init(entity='rafael-sterzinger', project='few_shot_map_seg', id=id if args.exp_name else None)

    fix_randseed(args.seed)

    log_info(args)
    wandb.config.update(args)
    
    experiment(args)
