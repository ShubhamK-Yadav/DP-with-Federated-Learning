
# Differential Privacy in Federated Learning Pipeline for Prostate Cancer Detection (PiCAI)

This repository contains the code to train a federated learning model using [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) for prostate cancer detection with Differential Privacy. The setup follows a semi-supervised federated learning scheme over 2 simulated clients.

### 1. Setup Environment
Make sure you are in a python virtual environment using the appropriate Python module and set your `PYTHONPATH`:

```bash
source ~/env/prostate-env/bin/activate
pip install --upgrade pip
module load Python/3.10.8-GCCcore-12.2.0
export PYTHONPATH=/users/aca21sky/prostate/prostate_2D
```

### 2. Install the Requirements
Navigate to the project directory and install dependencies:
```bash
cd prostate
pip install -r flare_requirements.txt
```
### 3. Preprocessing and Classification
Follow instructions in the original [ITUNet-for-PICAI-2022-Challenge](https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge/tree/main) to run preprocessing and classification steps. These are prerequisites before proceeding to FL training.

### 4. Generate Federated Client Splits
Run the following script to split your data among clients:
```bash
python generate_split.py --num_clients 2 --data_path /path/to/data
```
Arguments:
-- num_clients: Number of simulated clients (default = 5)
-- data_path: Path to the local dataset directory

### 5. Configure Federated Learning Job
Structure of the project is shown below, customise the config 
prostate/prostate_2D/job_configs/picai_fedsemi/
├── app/
│   └── config/
│       ├── config_fed_client.json
│       └── config_fed_server.json
└── meta.json

Update these files to:
-- Adjust number of clients, learning rate, epochs, etc.
-- Match client site names in meta.json with the expected setup.

### 6. Run FL Simulation
To start the FL simulation using NVIDIA FLARE from within prostate/prostate_2D/job_configs/picai_fedsemi/:
```bash
nvflare simulator . \
  -w ./workspace_picai_fedsemi \
  -n 2 \
  -t 2 \
  -gpu 0
```

Options:

-- -w: Path to the workspace
-- -n: Number of clients
-- -t: Number of training rounds
-- -gpu: GPU to use for training

### 6. Inference: Detection Phase
After training completes, you’ll get FL_global_model.pt from the server within the workspace created during the NVIDIA Flare simulation. Use this model for the inference below:
```bash
python inference_seg_fl.py \
  --workspace path/to/workspace \
  --test_dir /path/to/nnUNet_test_data \
  --output_dir /path/to/output_dir

```
This generates predictions needed for the detection phase of the challenge. This script retrieves the final output model from the simulation. Use the output for the detection phase mentioned in [ITUNet-for-PICAI-2022-Challenge](https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge/tree/main/segmentation). The predict_2d.py in the detection phase is replaced by the inference_seg_fl.py above. 

Results can be found in: https://docs.google.com/spreadsheets/d/1wgrYRIKwuNpygzsmzGCRTxYIjiLHE7zkHQ2r2-lQkQA/edit?usp=sharing
