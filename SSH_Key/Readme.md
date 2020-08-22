# SSH Key

## Mac Users

1. Open your terminal while the Xquartz is running.

2. Type the following command in your terminal.

```
ssh-keygen -m PEM -f ~/.ssh/gkey -C ubuntu
```

3. Hit enter and change your directory.

```
cd ~/.ssh
```

4. Check the list of the files under your directory.

```
ls
```

5. You should be able to see at least 2 files gkey and gkey.pub

6. Enter the following command to see your public key code.

```
cat gkey.pub
```

7. Copy the public code for configuring your GCP dashboard.

8. Now you need to configure the GCP dashboard and the instructions are in the videos that I provided.

9. Finally, in your terminal while you are in the same directory (/.ssh) enter the follwoing command to connect to your VM that you configured.

```
ssh -X -i gkey ubuntu@<External IP address from your Dashbaord VMs>
```

## Windows Users

1- Install [puttygen](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)

## Possible Errors

1. If you get Warning "REMOTE HOST IDENTIFICATIN HAS CHANGED" do the following:
```
nano know_hosts 
```

2. the window will open up remove the line which has your ip address that is confilicting (you can delete line by Ctrl+k)

3. Then do Ctrl +x type y and hit enter.

4. Redo the ssh and it should work.





## Softwares for connection to cloud:

* The following software's needs to be installed for connecting local machine to VM that you created in the cloud.


1. Mac users: Install [Xquartz](https://www.xquartz.org/), then reboot your computer.

2. Windows users: Install [Mobaxterm](https://mobaxterm.mobatek.net/download-home-edition.html), the installer version