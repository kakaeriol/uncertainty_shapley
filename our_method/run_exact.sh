data=$1
out_dir=$2
model=$3
kernel=$4
device=$5
for i in {0..10}; 
    do 
        echo $i
        python main.py --data ${data} --device ${device} --embd --n_projections 100  --output_dim 2 --model_aggregate ${model} --training_iteration 100 --kernel ${kernel} --n_random 0  --n_active 0 --out_dir ${out_dir} --seed $i 
    done
