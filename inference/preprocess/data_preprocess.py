import boto3

import urllib.parse
import io

from PIL import Image
import numpy as np
import cv2

from dynamic_data_transformer import DynamicDataTransformer

INFERENCE_NON_PROCESSED_BUCKET_NAME = 'smokynet-inference-images-non-processed'
INFERENCE_PROCESSED_BUCKET_NAME = 'smokynet-inference-images-processed'


def data_proprocessing(img_np_array):
    '''
    Preproceses the image represented in a np array and return it (tiled)
    '''
    ddt = DynamicDataTransformer()
    return ddt.transform(img_np_array)

def lambda_handler(event, context):

    s3 = boto3.client('s3')

    # extracting the bucket_name and the key of the new object that just 
    # invoked the trigger
    bucket_name_of_invoking_object = event['Records'][0]['s3']['bucket']['name']
    invoking_object_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'],
                                                    encoding='utf-8')

    # exception handling (should pass though for our use cases)
    try:
        response = s3.get_object(Bucket=bucket_name_of_invoking_object,
                                 Key=invoking_object_key)
        print("CONTENT TYPE: " + response['ContentType'])
        print(response)
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(invoking_object_key, bucket_name_of_invoking_object))
        raise e

    # get our object
    response_body = s3.get_object(Bucket=bucket_name_of_invoking_object, 
                                  Key=invoking_object_key)['Body']

    # read() returns a stream of bytes
    img_bytes = response_body.read()

    # convert the image stream into a np array
    img_np_array = cv2.imdecode(np.asarray(bytearray(img_bytes)), cv2.IMREAD_COLOR)

    # Send procesed image tiles to another s3 bucket
    processed_img_np_array_tiles = data_proprocessing(img_np_array=img_np_array)
    num_tiles = (processed_img_np_array_tiles.shape)[0]
    for i in range(num_tiles):

        # https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
        processed_img = Image.fromarray((processed_img_np_array_tiles[i] * 255).astype(np.uint8))

        # Save the image to an in-memory file
        in_mem_file = io.BytesIO()
        if processed_img.mode != 'RGB':
            processed_img = processed_img.convert('RGB')
        processed_img.save(in_mem_file, format='JPEG')
        in_mem_file.seek(0)

        # Upload image to s3

        processed_key = invoking_object_key # blah-blah.jpg
        processed_key = processed_key.replace('jpg', '') # blah-blah
        processed_key += '_processed' # blah-blah_processed
        processed_key += '/tile_' + str(i) + '.jpg' # blah-blah_processed/tile'i'.jpg where 'i' is the current index

        s3.upload_fileobj(
            Fileobj=in_mem_file,
            Bucket=INFERENCE_PROCESSED_BUCKET_NAME,
            Key=processed_key,
        )


