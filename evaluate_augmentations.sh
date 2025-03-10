python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --batch_size 8 --exp_name vanilla
python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --batch_size 8 --mixup --exp_name mixup
python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --batch_size 8 --cutmix --exp_name cutmix
python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --batch_size 8 --cutmix --mixup --exp_name mixup_cutmix