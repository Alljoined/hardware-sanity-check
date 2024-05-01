# stimulus-eeg

## Setup

1. create a `.env` file based off of [.env.template](./.env.template)
1. create a file called `cert.pem` containing the contents from [this](https://github.com/Emotiv/cortex-example/blob/master/certificates/rootCA.pem)
1. install all [requirements](./requirements.txt)
1. download `coco_file_224_sub1_ses1.mat` and put it into a folder called `processed-stimulus`
1. install the [emotive launcher](https://www.emotiv.com/products/emotiv-launcher#download)
1. log into the emotiv launcher. this is a background app that acts as a wss server
1. run [`stimulus.py`](./stimulus.py)
1. on the first time running this script, use the emotiv launcher to accept access from the headset

## Imagery

You will need to run download_coco.py to download the images first. This will take 22gb.
