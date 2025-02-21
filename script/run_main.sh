data_path="/external1/nguyenpham/code/uncertainty_shapley/data/mnist_non_iid_6.pickle"
out_dir="/external1/nguyenpham/code/uncertainty_shapley/output"

# ./run_random.sh ${data_path} ${out_dir} Net Exponential_SW_Kernel 0 2 > log/log_swel &
# ./run_random.sh ${data_path} ${out_dir} Net My_OTDD_SW_Kernel 1 0 > log/log_otdd &
# ./run_random.sh ${data_path} ${out_dir} Net base 2 0 > log/log_gpbase &
# ./run_exact.sh ${data_path} ${out_dir}  Net Exponential_SW_Kernel 3 0 > log/log_exact &

# ./run_active.sh ${data_path} ${out_dir} Net Exponential_SW_Kernel 0 2 > log/log_swel_a &
# ./run_active.sh ${data_path} ${out_dir} Net My_OTDD_SW_Kernel 1 0 > log/log_otdd_a &
# ./run_active.sh ${data_path} ${out_dir} Net base 2 0 > log/log_gpbase_a &
./run_nn.sh ${data_path} ${out_dir} Net 3 0 > log/log_nnbase &
