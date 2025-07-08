#!/usr/bin/env bash
# Optimized setup script for faster builds
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR

BASEDIR=$(dirname "$0")
mkdir -p "$BASEDIR"/app/libs

# Set optimization flags for workshop environment
export CMAKE_JOBS=16
export GRADLE_MAX_WORKERS=16
export ANDROID_ABIS=arm64-v8a  # Build only arm64 for workshop
export SKIP_TESTS=0  # Set to 1 to skip tests and save ~1-2 minutes

# Use the optimized build script
bash "$BASEDIR"/../../../../scripts/build_android_library_optimized.sh

cp "$BUILD_AAR_DIR/executorch.aar" "$BASEDIR"/app/libs/executorch.aar
