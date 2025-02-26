#!/bin/bash

set -e

# get path to this script
here=$(dirname $(readlink -f $0))
cd $here

export TESSERACT_API_PATH=tesseract_api.py

tesseract-runtime apply '{"inputs": {"data": "@testdata/*.json"}}' --output-format json+base64 | jq
tesseract-runtime jacobian '{"inputs": {"data": "@testdata/*.json"}, "jac_inputs": ["data.[0]"], "jac_outputs": ["data_sum"]}' | jq
