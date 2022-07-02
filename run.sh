git clone "https://github.com/F-Madruga/InceptionTime.git"
git clone "https://github.com/F-Madruga/ResNet.git"
git clone "https://github.com/F-Madruga/visual_transformer.git"
mv InceptionTime/src/* src
mv ResNet/src/* src
mv visual_transformer/src/* src
rm -rf InceptionTime/
rm -rf ResNet/
rm -rf visual_transformer/
python3 src/main.py