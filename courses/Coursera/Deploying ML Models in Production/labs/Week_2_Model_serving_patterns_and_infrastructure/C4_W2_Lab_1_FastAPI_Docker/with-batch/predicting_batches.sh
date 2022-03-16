#!/bin/bash
for (( i = 0; i < 3; i++ )); do
    echo -e "\nPREDICTING batch_${i}.json"
    curl -X POST http://localhost:81/predict \
    -d @./wine-examples/batch_${i}.json \
    -H "Content-Type: application/json"
done