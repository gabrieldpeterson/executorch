# Step 8: Testing on an Android Phone (Optional)
## Run on local machine
===

The following steps will be ran on a local terminal, and not in the Instruqt platform.

Make sure the Android device is connected
```bash
adb devices
```

Connect Android device via ADB
```bash
adb shell mkdir -p /data/local/tmp/llama
```

Copy all required files
```bash
adb push llama_main /data/local/tmp/llama
adb push llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte /data/local/tmp/llama
adb push tokenizer.model /data/local/tmp/llama
```

Make executable
```bash
adb shell chmod +x /data/local/tmp/llama/llama_main
```

Run with a prompt (following original tutorial format).
```bash
adb shell "cd /data/local/tmp/llama && ./llama_main --model_path llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte --tokenizer_path tokenizer.model --prompt 'You are Cookie, a helpful assistant. User: Hey Cookie, how are you today? Cookie:' --cpu_threads 5"
```
