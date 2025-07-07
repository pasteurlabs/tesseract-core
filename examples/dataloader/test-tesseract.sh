#!/bin/bash

set -e

# get path to this script
here=$(dirname $(readlink -f $0))
cd $here

# tesseract-run-label-begin
tesseract run dataloader \
    --volume $here/testdata:/mnt/testdata:ro \
    apply '{"inputs": {"data": "@/mnt/testdata/*.json"}}' | jq
# tesseract-run-label-end

tesseract run dataloader \
    --volume $here/testdata:/mnt/testdata:ro \
    jacobian '{"inputs": {"data": "@/mnt/testdata/*.json"}, "jac_inputs": ["data.[0]"], "jac_outputs": ["data_sum"]}' | jq
