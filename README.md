# IDS for DDoS detection with LSTM neural network.

This is prototype of IDS (Intrusion Detection System) with LSTM Reccurent Neural Network wich classificates incoming Network traffic and makes prediction to find DDOS attack in network packet's attributes. I have used that project as my final master project of Saint's Peterburg Polytechic University. 
The dataset was a Intrusion Detection Evaluation Dataset (CICIDS2017). For classification ddos trafic was used only a Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv. I can't public that dataset, but you can ask that University to share that dataset with you. 
This project only for stadying purposes, not for prodaction.

## Getting Started

You can just clone this repo to get a prepared CiCFlowMeter. 

### Prerequisites

You need to install python3 and pip3  
Also you need install the libpcap0.8-dev_1.8.1-6_amd64.deb.  

### Installing

The below commands will help you to install all needed dependencies:
```
sudo apt-get install python3-pip
sudo apt-get install tcpdump
sudo apt-get install libpcap-dev
sudo apt-get install inotify-tools
apt-get install python3-venv

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common

curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/debian \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo apt install default-jre
sudo apt install default-jdk
sudo apt-get install iptables-persistent

```
When you finished. You need to create and activate a virtual python environment:  
```
python3 -m venv <nameofenv>
source /<nameofenv>/bin/activate
```
After that you need to install python requirenments:

```
pip3 install -r requirenments.txt
```


## Deployment
For kafka brokers will be available you need to modify your /etc/hosts:
```echo 127.0.1.1 kafka >> /etc/hosts
```
Firstly you need run docker containers:

```
./run_compose.sh
```
After that you need to capture network packets:

```
cd tcpdump_and_cicflowmeter 
sudo ./capture_interface_pcap.sh <network interface> pcap <user>
```
Pickup csv files, and send data to kafka broker. You need do that in python virtual environment, in other terminal window.

```
./pickup_csv_files.sh csv/
```

Consume data from kafka:

```
cd python_client
python3 consumer.py
```

If you want to block ip address, wich will be find by neural network:


```
cd resources
rm black_list/black_list.txt
./block.sh black_list/
```


For demonstration and for more details, I recored video: https://www.youtube.com/watch?v=RiLg3jlOXjk&feature=youtu.be 

I have tested with LOIC: https://github.com/NewEraCracker/LOIC

## Built With

* Keras
* Kafka
* Docker
* Python3
* CiCFlowMeter
* Tcpdump
* Bash

## Contributing

Any pull requests are welcome.

## Authors

* **Artem Salagaev** - https://github.com/ArtemGits

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
