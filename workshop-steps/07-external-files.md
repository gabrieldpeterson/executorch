# Step 7: Build and Download Required Files

## Build the Android Extension AAR
===
The Android demo application requires AAR (Android Archive) files that contain the ExecuTorch runtime and JNI bindings. These need to be built before you can use the demo app.

Build the Android extension
```bash,run
pushd extension/android && ./gradlew build && popd
```

**Why**: This builds the Android Archive files that contain:
- ExecuTorch Android runtime
- JNI bindings for Java/Kotlin integration
- Native libraries for Android

## Run LlamaDemo Setup
===

```bash,run
pushd examples/demo-apps/android/LlamaDemo && ./gradlew :app:setup && popd
```
**Note**: At the end you will get a **BUILD FAILED** error, due to cmake version conflicts, but the resulting AAR files are sufficient for our needs

## Package All Required Files
===
Create a package with all files needed for the Android demo app.

Create a directory to house the files.
```bash,run
cd ~/chatbot/executorch
mkdir -p android_build_package
cd android_build_package
```
Copy AAR files
```bash,run
cp ../extension/android/executorch_android/build/outputs/aar/*.aar .
```

Copy JNI library
```bash,run
cp ../cmake-out-android-arm64-v8a/extension/android/libexecutorch_jni.so .
```

Copy llama_main executable
```bash,run
cp ../cmake-out-android/examples/models/llama/llama_main .
```

Copy model
```bash,run
cp ../llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte .
```

Copy tokenizer
```bash,run
cp ~/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model .
```

List files to verify everything has been copied.
```bash,run
ls -la
```
There should be five files: executorch_android-debug.aar, executorch_android-release.aar, libexecutorch_jni.so, llama_main, llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte, and tokenizer.model.


## Transfer Files to Local Machine
===

To transfer the built files to your local machine:

Create a compressed archive
```bash,run
cd /root/chatbot/executorch
tar -czf android_build_package.tar.gz android_build_package/
```

Download a copy of these files in a terminal on your local machine. These will be the exact same as the ones you just built.
```bash
wget https://armwearedevelopersws.blob.core.windows.net/publicfiles/android_build_package.tar.gz
```


