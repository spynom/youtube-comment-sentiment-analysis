#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 051826734860.dkr.ecr.ap-southeast-2.amazonaws.com

# Pull the latest image
docker pull 038462774337.dkr.ecr.us-east-1.amazonaws.com/spynom:latest

# Check if the container 'my-app' is running
if [ "$(docker ps -q -f name=my-app)" ]; then
    # Stop the running container
    docker stop my-app
fi

# Check if the container 'campusx-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=my-app)" ]; then
    # Remove the container if it exists
    docker rm campusx-app
fi

# Run a new container
docker run -d -p 80:5000 --name my-app 038462774337.dkr.ecr.us-east-1.amazonaws.com/spynom:latest