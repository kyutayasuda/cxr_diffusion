import argparse
from share import *

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyInference
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import numpy as np
import torch
import torchvision
from PIL import Image

def gen_cxr_path(prompt, file_name):
    resume_path = '/content/drive/My Drive/Uni/csit998/finetune_prompt2cxr.ckpt'

    batch_size = 1
    logger_freq = 1
    learning_rate = 1e-5
    sd_locked = False
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to(torch.device("cuda:0"))

    # create dataset
    x = torch.rand(1,256,256,3)
    batch = {'jpg':x,'txt':[prompt],'hint':x}
    log = model.log_images(batch)
    grid = torchvision.utils.make_grid(log['samples_cfg_scale_1.10'], nrow=4)
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.cpu().numpy()
    grid = (grid * 255).astype(np.uint8)
    path = f'./{file_name}.png'
    Image.fromarray(grid).save(path)
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chest X-Ray Image from Prompt")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing prompts and file names")

    args = parser.parse_args()

    # Load the CSV file into a DataFrame
    df = pd.read_csv(args.csv_path)

    # Iterate over DataFrame rows and generate images
    for index, row in df.iterrows():
        prompt = row['findings']
        file_name = row['number']
        img_path = gen_cxr_path(prompt, file_name)
        print(f"Generated image saved at: {img_path}")
