data=$1
out_dir=$2
model=$3
kernel=$4
device=$5
dim=$6

for i in {0..10}; 
    do 
    for j in {5..25..5}; 
        do
            echo $i $j
            echo "python main.py --data ${data} --device ${device} --embd --n_projections 100  --output_dim ${dim} --model_aggregate ${model} --training_iteration 100 --kernel ${kernel} --n_random $j  --n_active 0 --out_dir ${out_dir} --seed $i "
            python main.py --data ${data} --device ${device} --embd --n_projections 100  --output_dim ${dim} --model_aggregate ${model} --training_iteration 100 --kernel ${kernel} --n_random $j  --n_active 0 --out_dir ${out_dir} --seed $i 
            
        done   
    done
