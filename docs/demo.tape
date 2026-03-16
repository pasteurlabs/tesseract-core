# Tesseract Core demo GIF
# Records a quick getting-started flow: install → init → download example → build → run
#
# Prerequisites:
#   - Docker running
#   - Python 3.10+ available
#   - VHS
#
# Run with: vhs docs/demo.tape

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

# Hidden setup: clean slate + fresh venv
Hide
Type "rm -rf /tmp/my-tesseract /tmp/demo-venv"
Enter
Sleep 1s
Type "python3 -m venv /tmp/demo-venv && VIRTUAL_ENV_DISABLE_PROMPT=1 source /tmp/demo-venv/bin/activate && pip install -q --upgrade pip tesseract-core && tesseract --help"
Enter
Sleep 10s
Type "mkdir /tmp/my-tesseract && cd /tmp/my-tesseract"
Enter
Type "clear"
Enter
Sleep 1s
Show

# --- Install ---

Type "# Install Tesseract"
Enter
Sleep 1s
Type "pip install -q tesseract-core"
Enter
Sleep 3s
Enter

# --- Init a new project ---

Type "# Create a new Tesseract project"
Enter
Sleep 1s
Type "tesseract init --name my-tesseract"
Enter
Sleep 2s
Enter

# --- Download a pre-built example ---

Type "# Download an example (vector addition with analytic Jacobian)"
Enter
Sleep 1s
Type "curl -so ./tesseract_api.py https://raw.githubusercontent.com/pasteurlabs/tesseract-core/main/examples/vectoradd/tesseract_api.py"
Enter
Sleep 2s
Enter

# --- Build ---

Type "# Build it into a container"
Enter
Sleep 1s
Type "tesseract build ."
Enter
Sleep 6s
Enter

# --- Run apply ---

Type "# Run it — vector addition with s*a + b"
Enter
Sleep 1s
Type `tesseract run my-tesseract apply '{"inputs": {"a": [1, 2, 3], "b": [10, 20, 30]}}'`
Enter
Sleep 3s
Enter
Enter

# --- Run jacobian ---

Type "# Compute the Jacobian — gradients out of the box"
Enter
Sleep 1s
Type `tesseract run my-tesseract jacobian '{"inputs": {"a": [1, 2, 3], "b": [10, 20, 30]}, "jac_inputs": ["a"], "jac_outputs": ["result"]}'`
Enter
Sleep 3s
Enter
Enter

Type "# No go and build your own!"
Enter
Sleep 10s
