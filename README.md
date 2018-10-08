# heartbeat-htm
Detecting ECG anomaly using HTM

## Depedencty
NuPIC.core: https://github.com/numenta/nupic.core <br>
xtensor: https://github.com/QuantStack/xtensor/

## Dataset
Please download the dataset from [here](https://www.kaggle.com/shayanfazeli/heartbeat) then extract the files to the project directory

## Build and run
```
c++ main.cpp -o main -O3 -std=c++14 -lnupic_core
./main
```
