This example can be executed using the Tesseract Python SDK or CLI.

# Python SDK Instructions
The code to run the example using the Python SDK can be found in `examples/multi-tesseracts/multi_helloworld/run_example.py`. To run the example, simply execute the Python script:
```
python run_example.py
```

# CLI Instructions

### 1. Build Tesseracts
First, we need to build the helloworld and multi-helloworld tesseracts:
```
tesseract build ./examples/helloworld
tesseract build ./examples/multi-tesseracts/multi-helloworld
```

### 2. Optionally Create Network
Creating a network with Docker is optional in this example. If we choose not to create a network and omit the `--network` argument in the following commands, the Tesseracts will be automatically connected to the `bridge` network.

We can create a network with:
```
docker network create my_network
```

### 3. Serve helloworld Tesseract
To serve the Tesseract, run:
```
tesseract serve "helloworld:latest" --no-compose --network my_network
```
This command will print relevant container metadata to `stdout`. Importantly, the container metadata tells us the Tesseract's IP address for each network it is connected to. In our case, the helloworld Tesseract is connected to `my_network` (or `bridge` if no network was created) and is reachable at `172.19.0.2:8000`.
```
 [i] Serving Tesseract at http://127.0.0.1:53385
 [i] View Tesseract: http://127.0.0.1:53385/docs
 [i] Container ID: 40290da47fec59e36a4066185bdd7ed5df3e73742d522ba2f56d907eca2e3e96
 [i] Name: recursing_kirch
 [i] Entrypoint: ['tesseract-runtime']
 [i] View Tesseract: http://127.0.0.1:53385/docs
 [i] Tesseract project ID, use it with 'tesseract teardown' command: recursing_kirch
{"project_id": "recursing_kirch", "containers": [{"name": "recursing_kirch", "port": "53385", "ip": "127.0.0.1", "networks": {"my_network": {"ip": "172.19.0.2", "port": 8000}}}]}%
```

### 4. Run multi-helloworld Tesseract
With the helloworld Tesseract served, we can now run the multi-helloworld Tesseract:
```
tesseract run "multi-helloworld:latest" apply '{"inputs": {"helloworld_tesseract_url": "172.19.0.2:8000" , "name": "YOU"}}' --network my_network
```

### 5. Clean up
Finally, we need to tear down the helloworld tesseract. We can get the project/container ID from the metadata printed by the serve command or by running `tesseract ps` to see the list of currently served Tesseracts.
```
uv run tesseract teardown recursing_kirch
```
