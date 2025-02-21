data=$1
out_dir=$2
model=$3
device=$4
dim=$5


for i in {0..10}; 
    do 
        for j in {8..56..8}; 
            do
            echo $i $j
            python /external1/nguyenpham/code/Develop_uncertainty/our_method/main_nn.py --data ${data} --device ${device} --model_aggregate ${model} --training_iteration 100  --n_random $j --n_active 0 --out_dir ${out_dir} --seed $i    
            done
    done