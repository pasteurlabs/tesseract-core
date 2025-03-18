# Vectoradd (with torch)

This tesseract is similar to `vectoradd`, but returns a nested dictionary
with an entry for the vectors added and one for the vectors subtracted.
These both contain result and normed_result as outputs.
The tesseract uses pytorch behind the scenes. Also,
it implements all auto-diff endpoints except abstract-eval.
