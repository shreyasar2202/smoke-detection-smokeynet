# import json
# import boto3
 
# # Define the client to interact with AWS Lambda
# client = boto3.client('lambda')
 
# def lambda_handler(event,context):
 
#     # print(event.keys())
    
#     httpClient = boto3.client('sagemaker-runtime')
#     custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
#     endpoint_name = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/pytorch-inference-2022-11-20-01-43-13-677/invocations"                                       # Your endpoint name.
#     response = httpClient.invoke_endpoint(
#         EndpointName=endpoint_name, 
#         CustomAttributes=custom_attributes, 
#         ContentType=content_type,
#         Accept=accept,
#         Body=payload
#         )

#     print(response)  
 
#     # response = client.invoke(
#     #     FunctionName = 'arn:aws:lambda:us-west-2:113998783017:function:end-user-lambda',
#     #     InvocationType = 'RequestResponse',
#     #     Payload = json.dumps(event)
#     # )
 
#     # responseFromChild = json.load(response['Payload'])
#     # responseFromChild["ack1"] = "Ack1 payload"
#     # responseFromChild["has_img"] = event.get("img_data") is not None
 
#     # print('Response from child \n')
#     # print(responseFromChild)
#     # return responseFromChild


import boto3
import json
import time

client = boto3.client('lambda')
def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))
    
    # data = json.loads(json.dumps(event))
    data = json.dumps(event)
    print(data)
    # payload = data['data']
    # print(payload)
    #Getting entry in dynamodb
    flag = 0
    dynamodb = boto3.resource('dynamodb')
    #table name
    table = dynamodb.Table('fire-detections')
    response = table.get_item(Key={'camera_name': event['camera_name']})
    if 'Item' in response:
         print(response['Item'])
         timestamp = int(response['Item']['timestamp'])
         if(timestamp-86400>=0):
             print("Invoking endpoint")
             flag = 1
    else:
        flag = 1
        print('no item, so adding it to dynamodb and invoking endpoint')
        response = table.put_item(
  Item={
        'camera_name': event['camera_name'],
        'timestamp': event['timestamp'],
    }
    print("Succesfully uploaded the camera data to dynamodb",response)
    
    result = "Not invoking endpoint"
    if flag == 1:
        response = client.invoke(FunctionName='arn:aws:lambda:us-west-2:113998783017:function:end-user-lambda',
                                           InvocationType = 'Event',
                                           Payload = data)
        print(response)
        # result = json.loads(response)
        # result = response 
        result = json.loads(response['Payload'].read().decode('utf-8'))
        result["loc_smoke_ack"] = str(time.time())
        print(result)
    
    return result