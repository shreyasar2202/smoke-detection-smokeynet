import os
import json

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role, Session



sess = Session()

role = get_execution_role()
model_dir = "/home/ec2-user/SageMaker/smoke-detection-smokeynet/src" # Replace with S3

model = PyTorchModel(
    entry_point="inference.py",
    source_dir=model_dir,
    role=role,
    model_data=model_dir+"/last.tar.gz",
    framework_version="1.5.0",
    py_version="py3",
)

from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


local_mode = True

if local_mode:
    instance_type = "local"
else:
    instance_type = "ml.g4dn.xlarge"

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

import random
import numpy as np

dummy_data = {"inputs": np.random.rand(2, 45, 2, 3, 224, 224).tolist()}


res = predictor.predict(dummy_data)