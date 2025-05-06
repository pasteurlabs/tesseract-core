Tesseract Bootcamp
====================

````{note}
Complete this bootcamp to create and run your first very own Tesseract!
````

We at the Tesseract team have recently received a mysteriously encoded
message hidden in our lounge room.

Help us figure out what this message means -- in a Tesseract way.


```{toctree}
:caption: Tesseract Bootcamp
:maxdepth: 2
:hidden:

```

## Step 0: Initializing Your First Tesseract

Create your own Tesseract named `bootcamp` by running the command `tesseract init bootcamp`.

This will create a brand new Tesseract directory `bootcamp` with an empty api, config, and requirements file.

Run `tesseract build bootcamp` to validate everything is working.

:::{dropdown} Unit Test
:color: success
:icon: check-circle-fill

Run the test for the first step.

`pytest tutorial/test_bootcamp --dir [path to your tesseract directory]`

Make sure the first test is passing.
:::

## Step 1: Creating our own apply schema

````{note}

Learn how to define your own input/outpu schema and write your first apply function.

````
Change input/output schema.
Create apply function to convert input array into words.

:::{dropdown} Unit Test
:color: secondary
:icon: check-circle

`pytest tutorial/test_bootcamp -k "test_02_tesseract_apply"`
::::

## Step 2: Passing in the key

Pass in key as package_data.
Use the package data key to decode the message.


## Step 2a: Try out vectoradd

Try using vectoradd to see if you can combine results from
2 tesseracts and get the result.
