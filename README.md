# textual-entailment-example

## Step 1
Build a model image using docker and test locally

## Step 2
Push the image to AWS ECR using /project/SageMaker/container/build_and_push.sh

## Step 3
Put model file (checkpoints and vocabulary) to AWS S3 bucket

## Step 4
Create SageMaker Model and Endpoint configuration on AWS console

## Step 5
Deploy the Lamdba package to AWS Lambda 
