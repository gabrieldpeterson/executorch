# Step 2: Install Android Studio & Android NDK

## Install Android Studio
===
Download Android Studio
```bash,run
cd /tmp
wget https://armwearedevelopersws.blob.core.windows.net/publicfiles/android-studio-2023.1.1.24-linux.tar.gz
```

Extract, then make the sym link
```bash,run
sudo tar -xzf android-studio-*.tar.gz -C /opt && sudo ln -s /opt/android-studio/bin/studio.sh /usr/local/bin/android-studio
```

**Why**: Android Studio provides the Android SDK, though we'll mainly use command-line tools.

## Create SDK directory and download command-line tools
===
```bash,run
mkdir -p ~/Android/cmdline-tools
cd ~/Android/cmdline-tools

wget https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip
```

Unzip and move into the `cmdline-tools` directory
```bash,run
unzip commandlinetools-linux-*.zip
mv cmdline-tools latest
```

**Why**: Command-line tools allow us to manage Android SDK components without the GUI.

## Set up environment variables
===
```bash,run
echo 'export ANDROID_HOME=$HOME/Android' >> ~/.bashrc
echo 'export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Why**: These environment variables help tools locate the Android SDK.

## Accept SDK licenses and install required packages
===
Accept the license agreements. Press 'y', then 'Enter', as many times as prompted.
```bash,run
sdkmanager --licenses
```

Install the required Android SDK components.
```bash,run
sdkmanager "platform-tools" \
           "platforms;android-34" \
           "build-tools;34.0.0" \
           "ndk;28.0.12433566"
```

**Why**:
- `platform-tools`: ADB and other essential Android tools
- `platforms;android-34`: Android API level 34 (Android 14)
- `build-tools`: Tools for building Android apps
- `ndk;28.0.12433566`: Native Development Kit for C++ development (specific version for compatibility)

## Set NDK environment variable
===
```bash,run
echo 'export ANDROID_NDK=$ANDROID_HOME/ndk/28.0.12433566/' >> ~/.bashrc
echo 'export ANDROID_ABI=arm64-v8a' >> ~/.bashrc
source ~/.bashrc
```

**Why**: ExecuTorch build scripts need to know where the Android NDK is located.
