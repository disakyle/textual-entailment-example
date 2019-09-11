# -*- coding: utf-8 -*-
import json
import traceback
import re
import os
import signal
import time
import boto3
from datetime import datetime

client = boto3.client('sagemaker')
runtime_client = boto3.client('runtime.sagemaker')
dynamodb = boto3.client('dynamodb')
eventWatch = boto3.client('events')

def lambda_handler(event, context):
    
    # Check the SageMaker Endpoint status
    def get_endpoint_status():
        status = None
        try:
            endpoint_status = client.describe_endpoint(
                EndpointName='textual-entailment-endpoint'
                )
            status = endpoint_status['EndpointStatus']
        except:
            status = "notExist"
        return status
    
    # Initiate the SageMaker Endpoint
    def initiate_endpoint():
        client.create_endpoint(
            EndpointName='textual-entailment-endpoint',
            EndpointConfigName='textual-entailment-endpoint-configuration'
            )
        return
    
    # Delete the SageMaker Endpoint
    def terminate_endpoint():
        client.delete_endpoint(
            EndpointName='textual-entailment-endpoint'
            )
        return
    
    # Get the UTC time when the Endpoint is last invoked
    def get_last_invocation():
        item = dynamodb.get_item(
            TableName='GlobalItems',
            Key={'ProjectName':{'S':'textual-entailment'}}
            )
        last_invocation = item['Item']['lastInvocation']['S']
        last_invocation = datetime.strptime(last_invocation, '%Y-%m-%d %H:%M:%S.%f')
        return last_invocation
    
    # Update the UTC time of Endpoint last invocation 
    def update_last_invocation(new_time):
        dynamodb.update_item(
            TableName='GlobalItems',
            Key={'ProjectName':{'S':'textual-entailment'}},
            AttributeUpdates={'lastInvocation':{'Action':'PUT','Value':{'S':new_time}}}
            )
        return
    
    # Create local log in /tmp directory
    def create_log():
        timestamp = str(datetime.utcnow())
        update_last_invocation(timestamp)
        log_text = {
            'lastInvocation': timestamp,
            'lastDynamoUpdate': timestamp
        }
        with open('/tmp/last_invocation.jsonl','w') as f:
            json.dump(log_text, f)
            f.write('\n')
            f.close()
        return
            
    # Update the last invocation in /tmp directory to reduce utilization of DynamoDB
    def local_last_invocation(new_time, mode):
        with open('/tmp/last_invocation.jsonl','r') as f:
            invocation_text = json.load(f)
            f.close()
        if mode == 1:
            invocation_text['lastDynamoUpdate'] = new_time
            with open('/tmp/last_invocation.jsonl','w') as f:
                json.dump(invocation_text, f)
                f.write('\n')
                f.close()
            return
        else:
            invocation_text['lastInvocation'] = new_time
            last_update = invocation_text['lastDynamoUpdate']
            invocation_time = datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S.%f')
            update_time = datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S.%f')
            time_diff = (invocation_time-update_time).total_seconds
            with open('/tmp/last_invocation.jsonl','w') as f:
                json.dump(invocation_text, f)
                f.write('\n')
                f.close()
            return time_diff/60
    
    # Check the time elapsed between current UTC time and Endpoint last invocation time
    def check_time_interval():
        current_time = datetime.utcnow()
        last_invocation = get_last_invocation()
        time_diff = (current_time - last_invocation).total_seconds()
        return time_diff/60
    
    # Run the prediction and update Endpoint last invocation time
    def run_prediction(input_json):
        response = runtime_client.invoke_endpoint(
            EndpointName='textual-entailment-endpoint',
            Body=input_json,
            ContentType='application/json',
            Accept='Accept'
            )
        current_time = str(datetime.utcnow())
        if os.path.isfile('/tmp/last_invocation.jsonl'):
            try:
                minutes_diff = local_last_invocation(current_time, 2)
            except:
                minutes_diff = 0
            if minutes_diff > 5:
                update_last_invocation(current_time)
                local_last_invocation(current_time, 1)
        else:
            create_log()
        result = response['Body'].read()
        return result.decode('utf-8')
    
    status = get_endpoint_status()
    
    # If at the time of scheduled checking, time elapsed is more than 40 minutes, then delete the Endpoint.
    if "CheckStatus" in event and event["CheckStatus"]:
        time_interval = check_time_interval()
        if time_interval > 40 and status == 'InService':
            terminate_endpoint()
            # Terminate the CloudWatch Event as well
            eventWatch.disable_rule(
                Name="CheckStatus"
                )
        return {
        "statusCode": 200
        }

    indexPage=None
    
    # If no Endpoint instance is running, proceed to create the Endpoint
    if status == "notExist":
        initiate_endpoint()
        create_log()
        # Enable the CloudWatch Event for automatic Endpoint shutdown
        eventWatch.enable_rule(
            Name="CheckStatus"
            )
    
    if status != "InService":
        page_to_load = "loading.html"
    else:
        page_to_load = "index.html"
        
    with open(page_to_load, "r") as f:
        indexPage = f.read()
        f.close()

    method = event.get('httpMethod',{}) 
    if method == 'GET':
        return {
            "statusCode": 200,
            "headers": {
            'Content-Type': 'text/html',
            },
            "body": indexPage
        }

    if method == 'POST':
        bodyContent = event.get('body',{}) 
        parsedBodyContent = json.loads(bodyContent)
        taskHypothesis = parsedBodyContent["hypothesis"]["0"]
        taskPremise = parsedBodyContent["premise"]["0"]
        thisTask = parsedBodyContent["task"]["0"]

        timeout = False
        # handler function that tell the signal module to execute
        # our own function when SIGALRM signal received.
        def timeout_handler(num, stack):
            print("Received SIGALRM")
            raise Exception("processTooLong")

        # register this with the SIGALRM signal    
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # signal.alarm(10) tells the OS to send a SIGALRM after 10 seconds from this point onwards.
        signal.alarm(10)

        input_text = {'sentence1': taskPremise, 'sentence2': taskHypothesis}
        json_input = json.dumps(input_text)

        try:
            jsonResponse = run_prediction(json_input)
        except Exception as ex:
            if "processTooLong" in ex:
                timeout = True
                print("Processing Timeout!")
        finally:
            signal.alarm(0)
            
        success_status = "Fail"
        if re.search(thisTask, jsonResponse):
            success_status = "Success"

        return {
        "statusCode": 200,
        "headers": {
        "Content-Type": "application/json",
        },
        "body": json.dumps({
            "jsonFeedback": jsonResponse,
            "successStatus": success_status
            })
        }


