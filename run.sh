git clone "https://github.com/F-Madruga/InceptionTime.git"
git clone "https://github.com/F-Madruga/ResNet.git"
mv InceptionTime/src/* src
mv ResNet/src/* src
rm -rf InceptionTime/
rm -rf ResNet/
python3 src/main.py