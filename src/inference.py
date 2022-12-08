import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from main_model import MainModel
import util_fns
from lightning_module import LightningModule
import boto3
import numpy as np
from io import BytesIO
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SmokeyNet(nn.Module):
    def __init__(self, model):
        super(SmokeyNet, self).__init__()
        self.model = model
    
    def forward(self, x,  bbox_labels=None):
        tile_outputs = None
        outputs = None
        for i, sub_model in enumerate(self.model.model_list):

            if outputs is None or i > 0:
                outputs, x = sub_model(x, bbox_labels=bbox_labels, tile_outputs=outputs)
        return outputs, x


def model_fn(model_dir): 
    #print("model_dir",mode_dir)
    
    with open(os.path.join(model_dir, "last.ckpt"), "rb") as f:
        checkpoint = torch.load(f)
        hparams = checkpoint['hyper_parameters']
    
    num_tiles_height, num_tiles_width = util_fns.calculate_num_tiles((hparams['resize_height'],hparams['resize_width']), hparams['crop_height'], (hparams['tile_size'],hparams['tile_size']), hparams['tile_overlap'])

    main_model = MainModel(
                 # Model args
                 model_type_list=hparams['model_type_list'],
                 pretrain_epochs=hparams['pretrain_epochs'],
                 intermediate_supervision=hparams['no_intermediate_supervision'],
                 error_as_eval_loss=hparams['error_as_eval_loss'],
                 use_image_preds=hparams['use_image_preds'],
                 tile_embedding_size=hparams['tile_embedding_size'],

                 tile_loss_type=hparams['tile_loss_type'],
                 bce_pos_weight=hparams['bce_pos_weight'],
                 focal_alpha=hparams['focal_alpha'], 
                 focal_gamma=hparams['focal_gamma'],
                 image_loss_only=hparams['image_loss_only'],
                 image_pos_weight=hparams['image_pos_weight'],
                 confidence_threshold=hparams['confidence_threshold'],

                 freeze_backbone=hparams['freeze_backbone'], 
                 pretrain_backbone=hparams['no_pretrain_backbone'],
                 backbone_size=hparams['backbone_size'],
                 backbone_checkpoint_path=hparams['backbone_checkpoint_path'],

                 num_tiles=num_tiles_height * num_tiles_width,
                 num_tiles_height=num_tiles_height,
                 num_tiles_width=num_tiles_width,
                 series_length=hparams['series_length'],
                 is_background_removal=hparams['is_background_removal'])
    
    lightning_module = LightningModule.load_from_checkpoint(
                               os.path.join(model_dir, "last.ckpt"),
                               model=main_model,
                               batch_size=hparams['batch_size'],
                               optimizer_type=hparams['optimizer_type'],
                               optimizer_weight_decay=hparams['optimizer_weight_decay'],
                               learning_rate=hparams['learning_rate'],
                               lr_schedule=hparams['use_lr_schedule'],
                               series_length=hparams['series_length'],
                               parsed_args=hparams)
    
    model = SmokeyNet(lightning_module.model)
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    s3_client = boto3.client("s3")
    assert request_content_type == "application/json"
    bucket_name = json.loads(request_body)["bucket_name"]
    file_name = json.loads(request_body)["file_name"]
    
    input_data = BytesIO()
    s3_client.download_fileobj(bucket_name, file_name, input_data)
    input_data.seek(0)
    input_array = pickle.load(input_data)
    data = torch.tensor(input_array, dtype=torch.float32, device=device)
    
    split_file_name = file_name.split('_')
    
    camera_name = split_file_name[0]
    new_timestamp= str(int(split_file_name[1])-60)
    old_file_name = camera_name + '_' + new_timestamp + '__processed.pkl'
    
    try:
        print(old_file_name)
        input_data = BytesIO()
        s3_client.download_fileobj(bucket_name, old_file_name, input_data)
        input_data.seek(0)
        input_array = pickle.load(input_data)
        old_data = torch.tensor(input_array, dtype=torch.float32, device=device)
    except:
        old_data = torch.clone(data).to(device)
    
    data = torch.concat([old_data, data], dim=2)
    

    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction

def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions[1].cpu().numpy().tolist()
    return json.dumps(res)


