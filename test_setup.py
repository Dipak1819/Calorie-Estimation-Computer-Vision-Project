"""
Setup Verification Script
Run this to check if all dependencies are installed correctly
"""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def main():
    print("=" * 60)
    print("Checking Python version...")
    print("=" * 60)

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ is recommended")
    else:
        print("✓ Python version is compatible")

    print("\n" + "=" * 60)
    print("Checking required packages...")
    print("=" * 60)

    packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]

    all_installed = True
    for module, name in packages:
        if not check_import(module, name):
            all_installed = False

    print("\n" + "=" * 60)
    print("Checking model file...")
    print("=" * 60)

    import os
    if os.path.exists('best_food_model.h5'):
        file_size = os.path.getsize('best_food_model.h5') / (1024 * 1024)
        print(f"✓ Model file found (Size: {file_size:.2f} MB)")
    else:
        print("✗ Model file 'best_food_model.h5' not found")
        print("  Please train the model using main.ipynb first")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if all_installed:
        print("✓ All required packages are installed!")
        print("\nYou can now run the application with:")
        print("  python app.py")
    else:
        print("✗ Some packages are missing.")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")

    print("=" * 60)

if __name__ == '__main__':
    main()
