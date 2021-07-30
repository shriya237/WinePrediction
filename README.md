# <a href="https://www.docker.com/" title="Docker"><img src="icons/docker.png" /></a> Wine Quality Prediction ML model in Spark over AWS

The model has been trained using Random Forest Classifier on multiple EC2 instances using AWS EMR Spark. The model trained is saved in local/S3 bucket and loaded back in action for prediction on any preferred environment.

## Technologies used

 * Python
 * PySpark with ML libraries
 * AWS EMR
 * Docker
 * AWS S3
  

## Analysis on the Wine quality data

Please refer to Wine_Quality_Analysis_Prediction.ipynb notebook which shows the analysis done on the wine quality data - for instance correlation between different entities vs wine quality and pleasant visualizations helping to select suitable ML model.

## Steps to train model on AWS EMR

1) Create a cluster

```bash
$ aws emr create-cluster \
    --name "Cluster" \
    --release-label emr-5.33.0 \
    --applications Name=Spark \
    --ec2-attributes KeyName='practice.pem' \
    --instance-type m5.xlarge \
    --instance-count 4 \
    --use-default-roles
```

2) One way to submit task to Spark is by connecting to master node using SSH (before that make sure the security group has port 22 open)

```bash
$ spark-submit wineTraining.py 
```
						
3) The model results are stored in S3 bucket which have been saved in my local as well.

## Predictions on the saved model  

wineTest.py file loads the model metadata and predicts on the csv data provided. Output of the prediction vitally includes F1-score, classification report, confusion matrix, accuracy score as well as test error.

### Dockerization !!!

Here to make the prediction run on any environment like AWS, Azure or standalone machine, Dockerfile is designed.

How to build the docker image using Dockerfile

```bash
$ docker build -f Dockerfile -t shriya237/winetest:latest .
$ docker images
```

Now to push the docker to docker hub. Before that make sure that you are logged to docker hub from command line.

```bash
$ docker push shriya237/winetest:latest
```

## Step Guide to test the application (with Docker)

Create your own image using Dockerfile or make a pull request to the below public image as follows:

```bash
$ docker pull shriya237/winetest:latest
```

Run a container using the above image in attached mode.

```bash
$ docker run --rm shriya237/winetest:latest
```

Thus it predicts the model from the validation dataset provided and removes the container when it exits.


## Step Guide to test the application (without Docker)

Need to run the steps present in Dockerfile manually on the environment to setup Spark. Then do a spark submit along with the path of the file to test.

```bash
$ spark-submit wineTest.py data/ValidationDataset.csv
```

Thank you !!!