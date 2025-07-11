# Step 6: Build Llama Main for Android (with Tokenizer Support)

## Configure build for llama_main with tokenizer support
===
The key difference from executor_runner is that llama_main includes full tokenizer support, allowing it to process text prompts directly without external tokenization.
```bash,run
rm -rf cmake-out-android/examples/models/llama
/root/chatbot/venv/bin/cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/home/ubuntu/chatbot/venv/bin/python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_USE_TIKTOKEN=ON \
    -DEXECUTORCH_ENABLE_TESTS=OFF \
    -DBUILD_TESTING=OFF \
    -Bcmake-out-android/examples/models/llama \
    examples/models/llama
```

**Why**: Configures build specifically for llama_main with:
- Uses the original tutorial approach (separate build directory)
- Enables tiktoken tokenizer support with `-DEXECUTORCH_USE_TIKTOKEN=ON`
- Disables tests to avoid gtest dependency issues
- Includes all required kernels and XNNPACK support

## Build the llama_main
===
```bash,run
/root/chatbot/venv/bin/cmake --build cmake-out-android/examples/models/llama -j16 --target llama_main --config Release
```

**Why**: Builds the llama_main target with full tokenizer support using 16 parallel jobs.

## Verify the build
===
```bash,run
ls -la cmake-out-android/examples/models/llama/llama_main
file cmake-out-android/examples/models/llama/llama_main
```
