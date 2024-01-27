## Git eval

eval "$(ssh-agent -s)"

ssh-add key path

git init

git config --global user.name <name>

git config --global user.email <email>

git remote add origin https://github.com/amir-jafari/Machine-Learning.git

git add .

git commit -m "Updated"

git push -f origin master

git pull -f origin master

git config --global credential.helper wincred

git config --global credential.helper store

git remote set-url origin https://github.com/amir-jafari/Deep-Learning.git