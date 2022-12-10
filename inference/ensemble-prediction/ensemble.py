# import json
# import boto3
# import time
 
# # Define the client to interact with AWS Lambda
# client = boto3.client('lambda')
# s3 = boto3.client('s3')

# def lambda_handler(event,context):
#    print(event)
    #     # get our object
    # response_body = s3.get_object(Bucket='smokynet-inference-images-processed', 
    #                               Key='1465063200_-02400._processed/tile_0.npy')['Body']

    # # read() returns a stream of bytes
    # img_bytes = response_body.read()

    # print(img_bytes)

    # convert the image stream into a np array
    # img_np_array = cv2.imdecode(np.asarray(bytearray(img_bytes)), cv2.IMREAD_COLOR)
 
    # Define the input parameters that will be passed
    # on to the child function
    # inputParams = {
    #     "ProductName"   : "img1",
    #     "img_data"      : str(img_bytes)
    # }
 

 
import json
from pyproj import Proj, transform
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import boto3
import pandas as pd
from io import StringIO

def is_in_camera_direction(camera_geometry_pt, direction, wfabba_geometry_pt):
    if direction == "north":
        return wfabba_geometry_pt.y >= camera_geometry_pt.y
    elif direction == "south":
        return wfabba_geometry_pt.y <= camera_geometry_pt.y
    elif direction == "east":
        return wfabba_geometry_pt.x >= camera_geometry_pt.x
    elif direction == "west":
        return wfabba_geometry_pt.x <= camera_geometry_pt.x
    else:
        # unknown or something else 
        pass

def matches_distance_prox(camera_geometry, direction, radius_miles, wfabba_df):    
    wfabba_df["distance_m"] = wfabba_df["geometry"].distance(camera_geometry)
    wfabba_df["distance_mi"] = wfabba_df["distance_m"]/1609.344        
    match_results_df = wfabba_df[(wfabba_df["distance_mi"] <= radius_miles)].copy()
    
    if(len(match_results_df)>0):
    #filter for detections within same direction
        match_results_df["is_in_direction"] = match_results_df.apply(
            lambda row: is_in_camera_direction(camera_geometry, direction, row["geometry"]), axis=1
        )
        match_results_df = match_results_df[match_results_df["is_in_direction"] == True]

    return match_results_df

def lambda_handler(event, context):
    # TODO implement
    print("Event is : ", event)
    #print(" image id : ", json.loads(event['body'].get('image_id')))
    goes16_df = gpd.read_file('s3://goes-satellite-data/goes16_condensed.geojson')
    goes17_df = gpd.read_file('s3://goes-satellite-data/goes17_condensed.geojson')

    # df = pd.read_csv('s3://goes-satellite-data/camera_metadata_hpwren.csv')
    client = boto3.client('s3')

    bucket_name = 'goes-satellite-data'
    
    object_key = 'camera_metadata_hpwren.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    
    camera_metadata_df = pd.read_csv(StringIO(csv_string))
    
    camera_info = camera_metadata_df[camera_metadata_df['image_id']==event['image_id']].iloc[0]
    camera_latitude = camera_info['lat']
    camera_longitude = camera_info['long']
    camera_direction = camera_info['direction']
    timestamp = event['timestamp']
    smokeynet_timestamp = pd.Timestamp(timestamp, unit='s')
    radius_miles = 50
    time_buffer = 3600 * 2
    
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:3310')
    lat,lon = transform(inProj,outProj,camera_longitude,camera_latitude)
    camera_geometry = Point(lat, lon)
    print(camera_geometry)
    matched_goes16_df = matches_distance_prox(camera_geometry, camera_direction ,radius_miles, goes16_df)
    matched_goes17_df = matches_distance_prox(camera_geometry, camera_direction ,radius_miles, goes17_df)
    
    matched_goes16_df = matched_goes16_df[(matched_goes16_df['Timestamp'] < smokeynet_timestamp) & (matched_goes16_df['Timestamp'] >= smokeynet_timestamp - pd.Timedelta(seconds=time_buffer))]
    matched_goes17_df = matched_goes17_df[(matched_goes17_df['Timestamp'] < smokeynet_timestamp) & (matched_goes17_df['Timestamp'] >= smokeynet_timestamp - pd.Timedelta(seconds=time_buffer))] 
    
    goes16_prediction = 1 if len(matched_goes16_df) else 0
    goes17_prediction = 1 if len(matched_goes17_df) else 0
    
    votes = goes16_prediction + goes17_prediction + event['smokeynet_prediction']
    
    # majority voting
    
    if votes>=1:
        #call_lambda
        print("Number of votes : ", votes)
        client = boto3.client('lambda')
        response = client.invoke(
        FunctionName = 'arn:aws:lambda:us-west-2:113998783017:function:locating-smoke',
        InvocationType = 'Event',
        Payload = json.dumps({
            "camera_name" : event['image_id'],
            "latitude": camera_latitude,
            "longitude": camera_longitude,
            "timestamp": timestamp
        })
    )
 
    # result = json.loads(response['Payload'].read().decode('utf-8'))
    # result["ensemble"] = str(time.time())
    # print(result)   
    # return result
    
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }




