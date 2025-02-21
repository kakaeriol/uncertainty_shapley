# DUPRE: Data Utility Prediction for Efficient Data Valuation
This repository contains the official implementation of DUPRE, a framework designed to efficiently compute data valuations (e.g., Shapley values) by predicting coalition utilities with a Gaussian Process (GP) model. DUPRE supports any cooperative game–based data valuation (Shapley, Banzhaf, etc.) while reducing the need for expensive model retraining.

The framework display as following:


<img width="703" alt="image" src="https://github.com/user-attachments/assets/ffbf6404-7cf2-4a7c-b0a8-abf4282697a9" />


## Author
- Nguyen Pham
- Rachael Hwee Ling Sim
- Quoc Phong Nguyen
- See-Kiong Ng
- Bryan Kian Hsiang Low
## Repository Structure
```
├── README.md
├── notebook # where to demo the output/figure/image in our papers          
│   ├── 01_create_ds
│   ├── 00_fg2_intitution
│   └── 01_figure_02_01
├── our method # our package for GP methods which include our SSW and GP-binary kernel as well as NN-binary
│   ├── my_gpytorch # our gpytorch library
|   |   ├── kernels # some of our defined kernel here
|   |   ├── mymodels # the models that we defined.
│   └── 
└──  script # the example script to run our experiments                   
```
## Installation
1. Clone this repository:
```
git clone https://github.com/kakaeriol/dupre.git
cd dupre
```
2. (Optional) Create and activate a virtual environments
```
python -m venv venv
source venv/bin/activate  # For Linux/macOS
```
3. Install dependencies:
```
pip install -r requirements.txt
```
## Usage
1. Run experiment
```
python main.py --data ${data} --device ${device} --embd --n_projections 100  --output_dim ${dim} --model_aggregate ${model} --training_iteration 100 --kernel ${kernel} --n_random $j  --n_active 0 --out_dir ${out_dir} --seed $i
```
with ${data} is location for data, device: list of device gpu will be use, kernel can choose between "My_OTDD_SW_Kernel", "Exponential_SW_Kernel", "base", model_aggregate can choose between "Net", "MNIST_CNN", "ResNet_18_Classifier", "CNNRegressor", "MLPRegressor". 
To evaluate all of the ulitly function, can set n_random = n_active = 0. 
## Citation
If you use this code or data in your research, please cite:
```
```
