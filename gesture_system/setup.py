#!/usr/bin/env python3
"""
Setup script for Gesture Control Drone System
Python alternative to setup.sh for cross-platform compatibility
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description, show_output=False):
    """Run a command and display status."""
    print(f"  {description}...", end="", flush=True)
    try:
        if show_output:
            result = subprocess.run(command, shell=True, check=True)
        else:
            result = subprocess.run(command, shell=True, check=True, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        print(" ✓")
        return True
    except subprocess.CalledProcessError as e:
        print(f" ✗")
        print(f"    Error: {e}")
        return False


def main():
    print("=" * 50)
    print("Gesture Control Drone System Setup")
    print("=" * 50)
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print()
    print("Installing required packages...")
    print()
    
    # Upgrade pip first
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    # Core requirements
    packages = [
        ("opencv-python>=4.8.0", "OpenCV (computer vision)"),
        ("tensorflow>=2.13.0", "TensorFlow (deep learning)"),
        ("numpy>=1.24.0", "NumPy (numerical computing)"),
        ("Pillow>=10.0.0", "Pillow (image processing)"),
        ("mediapipe>=0.10.0", "MediaPipe (hand detection)"),
        ("matplotlib>=3.7.0", "Matplotlib (visualization)"),
        ("pandas>=2.0.0", "Pandas (data handling)"),
    ]
    
    # Install packages
    failed_packages = []
    for package, description in packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {description}"
        )
        if not success:
            failed_packages.append(description)
    
    # Try to install Crazyflie library
    print()
    print("Installing Crazyflie control library...")
    cflib_success = run_command(
        f"{sys.executable} -m pip install cflib>=0.1.25",
        "Installing cflib"
    )
    
    if not cflib_success:
        print("  ⚠ cflib installation failed - this is OK if you're only testing without a drone")
        print("  For drone control, you'll need to install libusb first:")
        print("    - Ubuntu/Debian: sudo apt-get install libusb-1.0-0")
        print("    - macOS: brew install libusb")
        print("    - Windows: Download from https://libusb.info/")
    
    # Test imports
    print()
    print("Testing imports...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("mediapipe", "MediaPipe"),
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            failed_imports.append(name)
    
    # Summary
    print()
    print("=" * 50)
    
    if failed_packages or failed_imports:
        print("⚠ Setup completed with some issues:")
        if failed_packages:
            print(f"  Failed to install: {', '.join(failed_packages)}")
        if failed_imports:
            print(f"  Failed to import: {', '.join(failed_imports)}")
        print()
        print("You may need to install these packages manually")
    else:
        print("✅ Setup completed successfully!")
    
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Make sure you have a trained model file 'best_model.keras'")
    print("   (This should be a Keras model trained on hand gesture images)")
    print()
    print("2. Test the gesture recognition system:")
    print(f"   python test_gesture_control_fixed.py")
    print()
    print("3. For actual drone control, you'll need:")
    print("   - A Crazyflie drone")
    print("   - Crazyflie radio dongle (Crazyradio PA)")
    print("   - libusb installed (see above)")
    print()
    
    # Check for model file
    if Path("best_model.keras").exists():
        print("✓ Found best_model.keras")
    else:
        print("⚠ No model file found. You'll need to:")
        print("  - Train a model on hand gesture data")
        print("  - Or download a pre-trained model")
        print("  - Save it as 'best_model.keras'")


if __name__ == "__main__":
    main()
