# Step 5: Build ExecuTorch and Libraries for Android

## Configure build with cmake
===
```bash,run
/root/chatbot/venv/bin/cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-23 \
  -DCMAKE_INSTALL_PREFIX=cmake-out-android \
  -DEXECUTORCH_ENABLE_LOGGING=1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
  -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON \
  -DXNNPACK_ENABLE_ARM_BF16=OFF \
  -Bcmake-out-android .
```

**Why**: This configures the build for:
- Android ARM64 architecture
- Minimum Android API 23 (Android 6.0)
- XNNPACK backend with KleidiAI optimizations for ARM
- Optimized and quantized kernels for efficient inference
- Required extensions for model loading and tensor operations

## Build and install
===
```bash,run
cmake --build cmake-out-android -j15 --target install --config Release
```

**Why**: Builds all components using 15 parallel jobs and installs them to the prefix directory.
