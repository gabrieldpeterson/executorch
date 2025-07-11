# Step 1: Initial Setup
## Requirements
===

This guide provides step-by-step instructions for building the ExecuTorch Llama runner for Android on a fresh Ubuntu 22.04 LTS instance.

## Prerequisites when replicating at home or in the office
This workshop will provide the instance to complete all the steps, minus the final ones where you upload the files from your personal machine to your Android phone.

When recreating these at home or in the office
* An Apple M1/M2 development machine with Android Studio installed or a Linux machine with at least 16GB of RAM.
* An Arm-powered smartphone with the i8mm feature running Android, with 16GB of RAM.
* A USB cable to connect your smartphone to your development machine.
* Android Debug Bridge (adb) installed on your device. Follow the steps in adb to install Android SDK Platform Tools. The adb tool is included in this package.
* Java 17 JDK. Follow the steps in Java 17 JDK to download and install JDK for host.
* Python 3.10.

## Install the necessary tools
===

Update the system and install all required development tools in one command:

```bash,run
sudo apt update && sudo apt install build-essential unzip openjdk-17-jdk python3-venv git-all cmake python3.10-dev -y
```

**Why**: These tools are essential for:
- `build-essential`: Compilers and build tools (gcc, g++, make)
- `unzip`: Extract downloaded archives
- `openjdk-17-jdk`: Required for Android Studio and Android build tools
- `python3-venv`: Create isolated Python environments
- `git-all`: Version control and submodule management
- `cmake`: Build system for C++ projects
- `python3.10-dev`: Python development headers needed for building Python extensions
