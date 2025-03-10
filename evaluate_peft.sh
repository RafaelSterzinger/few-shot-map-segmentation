for seed in 42 69 420
do
    #echo "python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --seed $seed --adapter none --exp_name none_$seed" >> gpu.queue
    echo "python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --seed $seed --adapter lora --exp_name lora_$seed" >> gpu.queue
    echo "python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --seed $seed --adapter loha --exp_name loha_$seed" >> gpu.queue
    echo "python run.py --class_name vineyard --base_model dino --nshots 10 --scale_factor 1 --epochs 50 --seed $seed --adapter lokr --exp_name lokr_$seed" >> gpu.queue
done
echo "Done"