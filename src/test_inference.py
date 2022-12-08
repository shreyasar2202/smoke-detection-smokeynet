import json
import os
import shutil
import tarfile

import boto3
import botocore
import numpy as np
import sagemaker

from inference import input_fn, model_fn, output_fn, predict_fn


def fetch_model(model_data):
    """Untar the model.tar.gz object either from local file system
    or a S3 location
    Args:
        model_data (str): either a path to local file system starts with
        file:/// that points to the `model.tar.gz` file or an S3 link
        starts with s3:// that points to the `model.tar.gz` file
    Returns:
        model_dir (str): the directory that contains the uncompress model
        checkpoint files
    """

    model_dir = "."
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

#     if model_data.startswith("file"):
#         _check_model(model_data)
#         shutil.copy2(
#             os.path.join(model_dir, "last.tar.gz"), os.path.join(model_dir, "last.tar.gz")
#         )
#     elif model_data.startswith("s3"):
#         # get bucket name and object key
#         bucket_name = model_data.split("/")[2]
#         key = "/".join(model_data.split("/")[3:])

#         s3 = boto3.resource("s3")
#         try:
#             s3.Bucket(bucket_name).download_file(key, os.path.join(model_dir, "last.tar.gz"))
#         except botocore.exceptions.ClientError as e:
#             if e.response["Error"]["Code"] == "404":
#                 print("the object does not exist.")
#             else:
#                 raise

    # untar the model
    tar = tarfile.open(os.path.join(model_dir, "last.tar.gz"))
    tar.extractall(model_dir)
    tar.close()

    return model_dir


def test(model_data):
    # model_dir = model_data
    model_dir = fetch_model(model_data)
    # load the model
    net = model_fn(model_dir)

    # simulate some input data to test transform_fn

    #data = {"inputs": np.random.rand(2, 45, 2, 3, 224, 224).tolist()}
    data = {'bucket_name' : 'smokynet-inference-images-processed', 'file_name':'bm-e-mobo-c_1670505095__processed.pkl'}
   

    serializer = sagemaker.serializers.JSONSerializer()

    jstr = serializer.serialize(data)
    jstr = json.dumps(data)

    content_type = "application/json"
    input_object = input_fn(jstr, content_type)
    predictions = predict_fn(input_object, net)
    res = output_fn(predictions, content_type)
    print(res)
    return

if __name__ == "__main__":
    model_data = "."
    test(model_data)