#!/bin/bash
# Build script specifically for Render deployment

echo "🚀 Starting Render build process..."

# Set environment variables for build
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PIP_NO_CACHE_DIR=1

# Upgrade pip and setuptools first to handle build_meta
echo "📦 Upgrading build tools..."
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools>=68.0.0 wheel

# Verify setuptools installation
echo "🔍 Checking setuptools installation..."
python -c "import setuptools.build_meta; print('setuptools.build_meta available')"

# Install requirements
echo "📦 Installing requirements..."
pip install --no-cache-dir -r requirements.txt

echo "✅ Render build completed successfully!"