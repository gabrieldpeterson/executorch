# Step 9: Build and Deploy the Chat App (Optional)
## Required files
===

The package you downloaded contains:
- **executorch_android-debug.aar** (28KB) - Debug version of the Android runtime
- **executorch_android-release.aar** (27KB) - Release version of the Android runtime
- **llama_main** (13MB) - Command-line executable for testing
- **llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte** (821MB) - Quantized model
- **tokenizer.model** (2.1MB) - Tokenizer for text processing
- **libexecutorch_jni.so** (233.9MB) - Java Native Interface bindings necessary for ExecuTorch communication

## Clone the app
===

Clone the repo into the same directory you extracted the above files to.
```bash
git clone https://github.com/gabrieldpeterson/executorch.git
```

## Open Android Studio
===

Open Android Studio, in the LlamaDemo project directory
```
(.../executorch/examples/demo-apps/android/LlamaDemo)
```

Update Gradle if prompted.

## Using the AAR Files in Android Studio
===

* In your terminal, if you aren't already there, navigate to all your files have been extracted to.
* In your Android project's `examples/demo-apps/android/LlamaDemo/app/` create a `libs` directory.
```bash
mkdir -p executorch/examples/demo-apps/android/LlamaDemo/app/libs
```
* Copy the AAR files to your Android project's `examples/demo-apps/android/LlamaDemo/app/libs` directory. Rename the "release" file so it is `executorch.aar`
```bash
cp executorch_android-release.aar executorch/examples/demo-apps/android/LlamaDemo/app/libs/executorch.aar
```
* Inside `LlamaDemo/app/src/main` create a `jniLibs/arm64-v8a/` directory
```bash
mkdir -p executorch/examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs/arm64-v8a
```
* Copy `libexecutorch_jni.so` into `LlamaDemo/app/src/main/jniLibs/arm64-v8a/` and rename it to `libexecutorch.so`
```bash
cp libexecutorch_jni.so executorch/examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs/arm64-v8a/libexecutorch.so
```
* In Android Studio, click Run at least once to initialize run-as access. You may need to sync with the Gradle files via File > Sync Project with Gradle Files before you can do so
* In your terminal, navigate to the uncompressed `android_buid_package` directory. Push the files to the expected directories on your Android device:

```bash
adb shell "run-as com.example.executorchllamademo mkdir -p files"

adb push llama_main /data/local/tmp/
adb shell "cat /data/local/tmp/llama_main | run-as com.example.executorchllamademo sh -c 'cat > files/llama_main'"

adb push tokenizer.model /data/local/tmp/
adb shell "cat /data/local/tmp/tokenizer.model | run-as com.example.executorchllamademo sh -c 'cat > files/tokenizer.model'"

adb push llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte /data/local/tmp/
adb shell "cat /data/local/tmp/llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte | run-as com.example.executorchllamademo sh -c 'cat > files/model.pte'"
```

* Verify they are in the correct location
```bash
adb shell "run-as com.example.executorchllamademo ls -lh files"
```
Back in Android Studio:
* File > Sync Project with Gradle Files
* Build > Rebuild Project
* Run
On your Android device:
* Click the Gear at the top right and select the model, tokenizer, and press "Load"

