# Transfering Files Using Command Lines

## Transferring Files in Amazon Web Services (AWS)


### First you need to open your computer terminal (Note: NOT THE VM TERMINAL)

* Uploading a file:
```
scp -i "your_aws_key.pem" [file-name] [aws-instance-name]:~/[file-name]
```

Example:
```
scp -i ~/Documents.aws.pem iris.csv ubuntu@ec2-52-207-225-47.compute-1.amazonaws.com:~/iris.csv
```
*  Downloading a file:
```
scp -i "your_aws_key.pem" [aws-instance-name]:~/ .
```
NOTE:->  .(dot) - Means current directory

Example:
```
scp -i ~/Documents.aws.pem ubuntu@ec2-52-207-225-47.compute-1.amazonaws.com:~/iris.csv .
```


## Transferring Files in Google Cloud Platform (GCP)


### First you need to open your GCP VM terminal (Note: NOT YOUR COMPUTER TERMINAL)
Note: You need to open your GCP dashboard and navigate to storage. Then create a bucket and name it. Here I created a test_bucket for this example:

*  Uploading a file:
```
gsutil cp [LOCAL_OBJECT_LOCATION] gs://[DESTINATION_BUCKET_NAME]/
```

Example:
```
gsutil cp -r /ajafari/test gs://test_buckect/
```
* Downloading a file:
```
gsutil cp gs://[BUCKET_NAME]/[OBJECT_NAME] [OBJECT_DESTINATION]
```

Example:
```
gsutil cp -r gs://test_buckect/amir.txt /home/ajafari/amir

```
# Transfering Files Using GUI Amazon Web Services (AWS) and Google Cloud Platform (GCP)


*  For Windows users you can use Mobaxterm and use SFTP option.
*  For Mac users you can use Filezila and use SFTP option.



