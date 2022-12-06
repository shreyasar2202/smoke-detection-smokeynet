import datetime
import json
import uuid
import time
import smtplib
import boto3

#Implementing a mail service to alert end-users
import json
import smtplib


def send_email(event):
    gmail_user = 'fire.alert.end.user@gmail.com'
    gmail_app_password = "jrbdwgdesounqsjp"
    sent_from = gmail_user
    sent_to = ['shivanihariprasad@gmail.com','ars.shreyas@gmail.com']
    timeval = str(time.ctime());
    sent_subject = "FIRE DETECTED"
    sent_body = "Fire has been detected at co-ordinates {}, {} at time {}".format(event['latitude'], event['longitude'], event['timestamp'])
    email_text = """\
From: %s
To: %s
Subject: %s
%s
""" % (sent_from, ", ".join(sent_to), sent_subject, sent_body)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_app_password)
        server.sendmail(sent_from, sent_to, email_text.encode("utf-8"))
        server.close()
        print(email_text)
        print('Email sent!')
    except Exception as exception:
        print("Error: %s!\n\n" % exception)


def lambda_handler(event, context):
    #Adding entry in dynamodb
    dynamodb = boto3.resource('dynamodb')
    #table name
    table = dynamodb.Table('fire-detections')
    response = table.update_item(
                Key={ "camera_name": event['camera_name']},
                UpdateExpression="set #timestamp=:r",
                ExpressionAttributeNames={
        '#timestamp': 'timestamp'
    
    },
                ExpressionAttributeValues={
        ':r': int(event['timestamp'])
        
    },
    ReturnValues="UPDATED_NEW"
            )
    
    print("Succesfully uploaded the camera data to dynamodb",response)

    
    
    send_email(event)
    return {
        'statusCode' : 200,
        'body': json.dumps('Hello from end-user lambda')
    }
    # # print("Recvd event" + json.dumps(event))
    # # return event['ProductName']

    # #1 Read the input parameters
    # #productName = event['ProductName']
 
    # #2 Generate the Transaction ID
    # transactionId   = str(uuid.uuid1())
 
    # #3 Implement Business Logic
    # print(transactionId)
    
 
    # #4 Format and return the result
    # return {
    #     'TransactionID' :   transactionId,
    #     'end_user_ack'  : str(time.time())
    # }
    # #     'ProductName'   :   productName,
    # #     'ack2'          :   "A payload",
    # #     "has_img_2"     :   event.get("img_data") is not None
    # # }
