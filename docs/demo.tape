# Tesseract Core demo GIF
# Records the full zero-to-running flow: init → edit → build → apply → jacobian
#
# Prerequisites:
#   - GNU nano (brew install nano) for syntax highlighting
#   - tesseract-core installed
#   - Docker running
#   - No existing /tmp/my-tesseract directory
#
# Run with: rm -rf /tmp/my-tesseract && vhs docs/demo.tape

Output "docs/img/demo.gif"

Set Shell "bash"
Set FontSize 14
Set Width 960
Set Height 550
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set WindowBar "Colorful"
Set WindowBarSize 40
Set BorderRadius 16
Set MarginFill "#ffffff"
Set Margin 8

# Hidden setup: nanorc for syntax highlighting + alias so we can type "nano"
Hide
Type "echo 'include /opt/homebrew/share/nano/python.nanorc' > /tmp/.demo-nanorc"
Enter
Sleep 300ms
Type "alias nano='/opt/homebrew/bin/nano --rcfile=/tmp/.demo-nanorc'"
Enter
Sleep 300ms
Type "clear"
Enter
Sleep 300ms
Show

# --- Init ---

Type "mkdir /tmp/my-tesseract && cd /tmp/my-tesseract"
Enter
Sleep 1s

Type "tesseract init --name my-tesseract"
Enter
Sleep 2s

# --- Edit tesseract_api.py in nano ---

Type "nano tesseract_api.py"
Enter
Sleep 1s

# 1. Add imports: search for "from pydantic", insert numpy above, runtime below
Ctrl+W
Type "from pydantic"
Enter
Sleep 300ms

Ctrl+A
Type "import numpy as np"
Enter
Sleep 300ms

Ctrl+E
Enter
Type "from tesseract_core.runtime import Array, Differentiable, Float64"
Sleep 300ms

# 2. Edit InputSchema: search for "pass", cut it, type comment + fields
Ctrl+W
Type "pass"
Enter
Sleep 300ms
Ctrl+K
Type "    # Two differentiable vector inputs"
Enter
Type "    a: Differentiable[Array[(None,), Float64]]"
Enter
Type "    b: Differentiable[Array[(None,), Float64]]"
Enter
Sleep 300ms

# 3. Edit OutputSchema: repeat search for next "pass"
Ctrl+W
Enter
Sleep 300ms
Ctrl+K
Type "    result: Differentiable[Array[(None,), Float64]]"
Enter
Sleep 300ms

# 4. Edit apply: search for "return OutputSchema()", replace
Ctrl+W
Type "return OutputSchema()"
Enter
Sleep 300ms
Ctrl+K
Type "    # Element-wise addition"
Enter
Type "    return OutputSchema(result=inputs.a + inputs.b)"
Enter
Sleep 300ms

# 5. Add jacobian function after apply
Enter
Type "# Analytic Jacobian: d(a+b)/da = I, d(a+b)/db = I"
Enter
Type "def jacobian(inputs, jac_inputs, jac_outputs):"
Enter
Type "    n = len(inputs.a)"
Enter
Type '    return {"result": {k: np.eye(n) for k in jac_inputs}}'
Enter
Sleep 1s

# Save and exit nano
Ctrl+O
Enter
Sleep 1s
Ctrl+X
Sleep 1s

# --- Build ---

Type "# Build the Tesseract container"
Enter
Sleep 1s
Type "tesseract build ."
Enter
Sleep 5s

# --- Run apply ---

Type "# Run the apply endpoint"
Enter
Sleep 1s
Type `tesseract run my-tesseract apply '{"inputs": {"a": [1, 2, 3], "b": [10, 20, 30]}}'`
Enter
Sleep 5s
Enter

# --- Run jacobian ---

Type "# Compute the Jacobian"
Enter
Sleep 1s
Type `tesseract run my-tesseract jacobian '{"inputs": {"a": [1, 2, 3], "b": [10, 20, 30]}, "jac_inputs": ["a"], "jac_outputs": ["result"]}'`
Enter
Sleep 5s
Enter

Sleep 2s
