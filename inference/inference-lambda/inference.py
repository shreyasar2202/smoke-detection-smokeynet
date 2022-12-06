import boto3
import json
import csv
import time

runtime= boto3.client('runtime.sagemaker')
client = boto3.client('lambda')

def lambda_handler(event, context):
    print(event)
    bucket_name_of_invoking_object = event['Records'][0]['s3']['bucket']['name']
    invoking_object_key = event['Records'][0]['s3']['object']['key']
    
    d = {}
    
    
    d['bucket_name'] = bucket_name_of_invoking_object
    # d['file_name'] = 'processed_data.pkl'
    d['file_name'] = invoking_object_key
    
    split_file_name = invoking_object_key.split('_')
    print("split_file_name : ", split_file_name)
    camera_name = split_file_name[0]
    timestamp = int(split_file_name[1])
    
    print(bucket_name_of_invoking_object)
    print(invoking_object_key)
                                                    
    #d = { 'bucket_name': "smokynet-inference-images-processed", 'file_name': 'processed_data.pkl'}
    print(d)
    s = json.dumps(d)
    print(s)
    endpoint_name = 'pytorch-inference-2022-11-25-03-26-51-996'

    runtime = boto3.Session().client(service_name='sagemaker-runtime',region_name='us-west-2')

    #response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=s)
    #smokeynet_prediction = json.loads(response['Body'].read())[0][1]
    
    smokeynet_prediction = 1  # Remove Later
    
    responseFromLambda = client.invoke(
    FunctionName = 'arn:aws:lambda:us-west-2:113998783017:function:ensemble-prediction',
    InvocationType = 'RequestResponse',
    Payload = json.dumps({
                 "image_id" : camera_name,
                 "smokeynet_prediction" : smokeynet_prediction,
                 "timestamp" : timestamp
                })
                        )
    result = json.loads(responseFromLambda['Payload'].read().decode('utf-8'))
    result["inference"] = str(time.time())
    print(result) 
    return result
    

 

    
    