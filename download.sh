FILE=$1

if  [ $FILE == "celeba-hq-dataset" ]; then
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-v2-dataset" ]; then
    #URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
    URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
    ZIP_FILE=./data/afhq_v2.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

else
    echo "Available arguments are celeba-hq-dataset, afhq-dataset and afhq-v2-dataset."
    exit 1

fi
