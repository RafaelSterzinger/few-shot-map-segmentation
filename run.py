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
from model.segmentation_head import SegmentationHead
import wandb
import argparse

from util import calculate_iou, calculate_objective, fix_randseed, log_info, print_trainable_parameters, to_cuda, calculate_f1, EarlyStopping

BASE_MODEL_DICT = {
    'dino' : 'facebook/dinov2-large',
    'radio' : 'nvidia/RADIO-L', #'nvidia/RADIO-H',
    'sam' : 'facebook/sam-vit-large',
    'apple' : 'apple/aimv2-large-patch14-224'
}



def run_epoch(model, dataloader, optimizer, scale_factor = 1):
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
            img = torchvision.transforms.functional.resize(img, [s*scale_factor for s in img.shape[-2:]],torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
            mask = batch['mask']

            nsamples += img.shape[0]

            if type(model) is Unet:
                f = model
                model = lambda x: f(x).squeeze(1)

            if is_train:
                optimizer.zero_grad()  # Zero out the gradients during training
                pred = model(img)
                loss = calculate_objective(pred, mask)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  

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
    if 'unet' not in args.base_model:
        base_model_path = BASE_MODEL_DICT[args.base_model]
        base_model = get_lora_model(base_model_path, args.adapter)
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

    dl_train, dl_val, dl_test = build_dataloader(args, preprocessor, args.mixup, args.cutmix)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    early_stopping = EarlyStopping(patience=10, min_delta=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.005)

    best_model = None
    best_iou = 0

    for epoch in range(args.epochs):
        train_loss, train_iou, train_f1 = run_epoch(model, dl_train, optimizer, args.scale_factor)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_miou": train_iou, "train_f1": train_f1})
        print(f"Epoch {epoch}, Loss: {train_loss:.4f}, mIoU: {train_iou:.4f}, F1: {train_f1:.4f}")
        val_loss, val_iou, val_f1 = run_epoch(model, dl_val, None, args.scale_factor)
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_miou": val_iou, "val_f1": val_f1})
        print(f"VALIDATION: Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}, F1: {val_f1:.4f}")
        # Reduce LR if validation IoU plateaus
        scheduler.step(val_iou)

        # Check early stopping (higher IoU is better)
        if early_stopping(val_iou):
            print("Early stopping triggered.")
            break

        if best_model is None or best_iou < val_iou:
            best_iou = val_iou
            best_model = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model)

    test_loss, test_iou, test_f1 = run_epoch(model, dl_test, None, args.scale_factor)
    wandb.log({"test_loss": test_loss, "test_miou": test_iou, "test_f1": test_f1})
    print(f"TESTING: Loss: {test_loss:.4f}, mIoU: {test_iou:.4f}, F1: {test_f1:.4f}")
    wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, choices=['railway', 'vineyard'], help="Chose railways or vineyards")
    parser.add_argument("--adapter", type=str, default='none', choices=['lora', 'lokr', 'loha', 'boft', 'none'], help="Low-rank adaptation methods")
    parser.add_argument("--base_model", type=str, choices=['dino', 'sam', 'radio', 'apple', 'unet'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--nshots", default=1.0)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument("--scale_factor", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()


    wandb.disabled = not args.exp_name
    random_string = ''.join(random.choices(string.ascii_lowercase, k=4))
    wandb.init(entity='rafael-sterzinger', project='few_shot_map_seg', id=f"{args.exp_name}_{random_string}" if args.exp_name else None)

    fix_randseed(args.seed)

    log_info(args)
    wandb.config.update(args)
    
    experiment(args)
