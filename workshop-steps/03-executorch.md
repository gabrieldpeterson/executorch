# Step 3: Set Up ExecuTorch

## Create project directory and Python virtual environment
===
Create a directory to place the chatbot files in.
```bash,run
cd
mkdir chatbot
cd chatbot
```

Create and activate a Python virtual environment.
```bash,run
python3 -m venv venv && source venv/bin/activate
```

## Clone ExecuTorch and initialize submodules
===
Clone the executorch repo and cd into it.
```bash,run
git clone https://github.com/gabrieldpeterson/executorch.git && cd executorch
```

Initialize the Git submodules
```bash,run
git submodule deinit -f . && git submodule sync --recursive && git submodule update --init --recursive
```

**Why**: ExecuTorch depends on several third-party libraries managed as Git submodules. These commands ensure all dependencies are properly initialized.

## Clean any stale cache and install ExecuTorch
===
Install ExecuTorch
```bash,run
./install_executorch.sh --clean && ./install_executorch.sh && ./examples/models/llama/install_requirements.sh
```
*Note: At the very end you'll see a message that says, "Failed to import examples.models due to lm_eval conflict." This is fine, and won't affect anything.*

**Why**:
- `--clean`: Removes any previous build artifacts
- The install scripts set up Python dependencies and build tools
- Llama-specific requirements include tokenizers and model utilities
