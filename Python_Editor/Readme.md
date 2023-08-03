# Pycharm Remote Editor

## Configuring Cloud Side:
1. SSH to your instance (AWS or GCP).

```
sudo git clone <the repo of deep learning>
```
2. As a test you can clone my Machine Learning repo.
```
sudo git clone https://github.com/amir-jafari/Machine-Learning.git
```

3. Change the repo folder permission to full access by 

```
sudo chmod -R 777 <name of the directory>
```
4. Set the display (enter the following command into command line):

```
export DISPLAY=localhost:10.0
```

## Pycharm Side:
1. Create a working directory in your computer (remember the path and location).
2. Open Pycharm Professional and on the main page click on create a new project.
3. Point the project to the directory that you created before (by browsing it).
4. Go to tools and deployments and configuration.
5. Click on the plus icon and name your remote deployment and the type should be SFTP.
6. Hit ok and then complete the host with the DNS or IP address that you got from cloud dashboard.
7. Fill the username as the one chose in your SSH key process (ubuntu).
8. Choose Authentication type as key pair then point it to your private key.
9. Test SFTP and you should get the successful connection.
10. On the second tab of deployment fill the mapping fields.
11. Fill the first path as the directory which you created in step 1 and second path is the cloud repo which you cloned it.
12. Go to tools and deployments download from the cloud.
13. Go to file and setting (preferences mac users) and search for project interpreter.
14. Click on the gear box and hit add.
15. On the left pan click on the SSH interpreter, choose the existing one (If you get the message on the window move or create choose move).
16. From the dropdown menu choose the one that you configured in the previous steps.
17. Select the path to the python (/usr/bin/python for python2 and /usr/bin/python3 for python3) 
18. Now click ok. You are done with the configuration.
19. Run one of the scripts from the repo.

### Possible errors:
1. If you get the waring no interpreter, please edit the environment (on top right when is the name of the script is) and set the interpreter to remote one.
2. If you get the error on the no file and such directory, then go to tools, deployment configuration and check all the SFTP configuration mapping (second tab) to local and cloud paths (Step 10 and 11).
3. For graphics and plots you need to set the display to local host 10.0. Edit the environment and click on the plus sign and add DISPLAY as name and localhost:10.0 as values hit OK.
4. Make sure in the environment the working directory is the one you are pointing at.
```
DISPLAY=localhost:10.0
```

### Caffe:

1. For caffe you need add extra environment field: 
```
PYTHONPATH=/home/ubuntu/caffe/python
```


## Extra Advance Options
```
import matplotlib
```
```
matplotlib.use('Qt4Agg')
```
```
import matplotlib
```
```
 matplotlib.use('TkAgg')
 ```

