data_path="/external1/nguyenpham/code/uncertainty_shapley/data/mnist_non_iid_5.pickle"
out_dir="/external1/nguyenpham/code/uncertainty_shapley/output"

./run_random.sh ${data_path} ${out_dir} Net Exponential_SW_Kernel 3 2 > mnist_3 &
./run_random.sh ${data_path} ${out_dir} Net My_OTDD_SW_Kernel 2 0 > mnist_2 &
./run_random.sh ${data_path} ${out_dir} Net Exponential_SW_Kernel 1 0 >> mnist_1 
./run_exact.sh ${data_path} ${out_dir} Net Exponential_SW_Kernel 0 0 > mnist_0 &

