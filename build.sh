#!/bin/bash
# Custom build script for Vercel deployment

echo "🚀 Starting custom build process..."

# Upgrade pip, setuptools, and wheel first
echo "📦 Upgrading build tools..."
python -m pip install --upgrade pip
pip install --upgrade setuptools>=65.0.0 wheel

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✅ Build completed successfully!"