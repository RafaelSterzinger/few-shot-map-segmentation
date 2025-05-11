#for seed in 42 69 420
#do
#    for type in railway vineyard
#    do
#        for nshots in 1.0
#        do
#            echo "python run.py --class_name $type --base_model radio_l --nshots $nshots --epochs 30 --batch_size 4 --scale_factor 3 --seed $seed --adapter lora --exp_name ours_final && sleep 3" >> star.queue
#        done
#    done
#done
#
for seed in 42 69 420
do
    for nshots in 0.01
    do
        echo "python run.py --class_name railway --base_model radio_l --nshots $nshots --epochs 200 --batch_size 4 --scale_factor 3 --seed $seed --adapter lora --exp_name ours_final && sleep 3" >> star.queue
    done
done

#for seed in 42 69 420
#do
#    for type in railway vineyard
#    do
#        for nshots in 10
#        do
#            echo "python run.py --class_name $type --base_model radio_l --nshots $nshots --batch_size 4 --scale_factor 3 --seed $seed --adapter lora --exp_name ours_final && sleep 3" >> gpu.queue
#        done
#    done
#done
#
## more seeds for smaller sample size
#for seed in 0 1 42 69 420
#do
#    for type in railway vineyard
#    do
#        for nshots in 5 1
#        do
#            echo "python run.py --class_name $type --base_model radio_l --nshots $nshots --batch_size 4 --scale_factor 3 --seed $seed --adapter lora --exp_name ours_final && sleep 3" >> gpu.queue
#        done
#    done
#done

echo "Done"