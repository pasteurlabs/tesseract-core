Tesseract Bootcamp
====================

````{note}
Complete this bootcamp to create and run your first very own Tesseract!
````

We at the Tesseract team have recently received a mysteriously encoded
message hidden in our lounge room.

:::{dropdown} Secret Message
```{literalinclude} ../../../tests/tutorial/example_data/secret_message.json
:language: json
```
:::


Help us figure out what this message means -- in a Tesseract way.

## Step 0: Initializing Your First Tesseract

Create your own Tesseract named `bootcamp` by running the command `tesseract init --name bootcamp --target-dir bootcamp`.

This will create a brand new Tesseract directory `bootcamp` with an empty api, config, and requirements file.

Run `tesseract build bootcamp` to validate everything is working.

:::{dropdown} Unit Test
:color: secondary
:icon: check-circle

Run the test for the first step.

* Make sure you run the build command first.

`pytest tests/tutorial/test_bootcamp.py --tesseract-dir [path to bootcamp directory] -k test_00_tesseract_init`

Example:
`pytest tests/tutorial/test_bootcamp.py --tesseract-dir bootcamp -k test_00_tesseract_init`

Make sure the first test is passing.
:::

## Step 1: Filling out the api

````{note}

Learn how to define your own input/output schema and write your first apply function.

````

Now, in your generated tesseract, there should be a semi-populated `tesseract_api.py` file.

Populate it by following the instructions here: [Get Started - tesseract_api.py](../../introduction/get-started.md#let-s-peek-under-the-hood).

For our `bootcamp` Tesseract, we want to be able to decode this encoded message that is a numpy array of random integers into a hidden string message.

**Populate the `input schema` and `output schema` fields** so we can achieve this, then make sure we pass the next test. You can reference the examples in `helloworld` and `vectoradd` of different ways to define the schema types. Also take note of how secret message is structured when defining your input types.

Make sure you rebuild the bootcamp Tesseract before proceeding to the testing phase.

:::{dropdown} Unit Test
:color: secondary
:icon: check-circle

`pytest tests/tutorial/test_bootcamp.py --tesseract-dir [path to bootcamp directory] -k test_01a_tesseract_schema`

example:
`pytest tests/tutorial/test_bootcamp.py --tesseract-dir bootcamp -k test_01a_tesseract_schema`

::::

Now that the schemas are populated, we need to fill out the apply function. We believe that the message is encoded using a basic *letter number cipher (A1Z26)*. **Write the apply function so that we can pass in our message and get the decoded result back.**

```{tip}
Use examples/vectoradd and examples/helloworld as inspo.
```

Test using the next unit test to validate we can decode some basic messages.


:::{dropdown} Unit Test
:color: secondary
:icon: check-circle

`pytest tests/tutorial/test_bootcamp.py -k test_01b_tesseract_apply`

You can also test this locally by running
```
tesseract run bootcamp apply @tests/tutorial/example_data/test_apply.json
```
::::

## Step 2: Passing in the key

````{note}

Learn how to pass in and process local data in your Tesseract.

````
Now, let's try passing in our secret message into the Tesseract to see if it can be correctly decoded.. Copy and paste the secret message above
into a file named `secret_message.json` and pass it into your Tesseract's
apply function!

```
tesseract run bootcamp apply @secret_message.json
```

Oh what's that? A bunch of gibberish? It seems that the decoder isn't quite correct... Hmmm...

It turns out, the janitor found a secret key stashed in the bathroom stall last night and we have reason to suspect that this key may help with the decoding.

:::{dropdown} Possible Key???
```{literalinclude} ../../../tests/tutorial/example_data/message_key.json
:language: json
```
:::

Create a file named `message_key.json` and follow the instructions in the [Package Data example](../../howto/packagedata.md) to read the message key into our apply function, and add the key to our input array before decoding to figure out the true message.

:::{dropdown} Unit Test
:color: secondary
:icon: check-circle

`pytest tutorial/test_bootcamp -k test_02a_tesseract_packagedata`
::::

And now we can finally reveal the secret message! Do you know what that's from?

##  Bonus (TODO)
### Step 2a: Try out vectoradd

Try daisy-chaining 2 Tesseracts together in python using vector-add and bootcamp tesseract.

### Step 3: Local Package

Convert the python script you just wrote using the Tesseracts into a local package.

### Step 4: ...
