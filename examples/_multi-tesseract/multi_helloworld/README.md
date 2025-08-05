# Multi-HelloWorld Example

## Build Tesseracts

First, we need to build the helloworld and multi-helloworld tesseracts:

```bash
$ tesseract build ./examples/helloworld
$ tesseract build ./examples/_multi-tesseract/multi-helloworld
```

### Create Network

Next, we create a network by running:

```bash
$ docker network create my_network
```

## Python API Instructions

This example can be executed using the Tesseract Python API or CLI. To run the example using the Python API, simply execute the Python script:

```bash
$ cd ./examples/_multi-tesseract/multi_helloworld/
$ python run_example.py
```

## CLI Instructions

### Serve `helloworld` Tesseract

To serve the Tesseract, run:

```bash
$ tesseract serve "helloworld:latest" --network my_network --network-alias helloworld
```

This command will print relevant container metadata to `stdout`. Importantly, the container metadata tells us the Tesseract's IP address for each network it is connected to. In our case, the helloworld Tesseract is connected to `my_network` and is reachable at `172.19.0.2:8000`. The `network-alias` argument allows us to create more human-readable addresses, in this example `--network-alias helloworld` also makes the Tesseract available at `http://helloworld:8000`.

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

### Run `multi-helloworld` Tesseract

With the helloworld Tesseract served, we can now run the multi-helloworld Tesseract:

```bash
$ tesseract run "multi-helloworld:latest" apply '{"inputs": {"helloworld_tesseract_url": "http://helloworld:8000" , "name": "YOU"}}' --network my_network
```

### Teardown

Finally, we tear down the helloworld tesseract using the container name from the metadata returned by the serve command (or by running `tesseract ps`):

```bash
$ tesseract teardown recursing_kirch
```
