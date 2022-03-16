#!/bin/bash

while sleep 0.005;

do curl -d '{"instances": [1.0, 2.0, 5.0, 7.0, 10.0, 100.0]}' -X POST $(minikube ip):30001/v1/models/half_plus_two:predict;

done
