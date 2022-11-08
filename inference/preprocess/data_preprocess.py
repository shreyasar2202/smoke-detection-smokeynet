import boto3

import urllib.parse
import io

from PIL import Image
import numpy as np
import cv2


INFERENCE_NON_PROCESSED_BUCKET_NAME = 'smokynet-inference-images-non-processed'
INFERENCE_PROCESSED_BUCKET_NAME = 'smokynet-inference-images-processed'


# TODO: Implement actual, instead of dummy
def data_proprocessing_dummy(img_np_array):
    '''
    Preproceses the image represented in a np array and return it

            Parameters:
                img_np_array (numpy.ndarray): the image represented in a np array
            Returns:
                processed_img_np_array (numpy.ndarray): processed veresion
    '''
    return img_np_array


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

    # Send procesed image to another s3 bucket
    processed_img_np_array = data_proprocessing_dummy(img_np_array=img_np_array)
    processed_img = Image.fromarray(processed_img_np_array)

    # Save the image to an in-memory file
    in_mem_file = io.BytesIO()
    processed_img.save(in_mem_file, format='JPEG')
    in_mem_file.seek(0)

    # Upload image to s3

    processed_key = invoking_object_key
    processed_key = processed_key.replace('.jpg', '_processed.jpg')
    s3.upload_fileobj(
        Fileobj=in_mem_file,
        Bucket=INFERENCE_PROCESSED_BUCKET_NAME,
        Key=processed_key,
    )


