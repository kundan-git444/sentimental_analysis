import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    "Flask",
    "pandas",
    "transformers",
    "scipy",
    "matplotlib",
    "nltk",
    "torch"  # Optional, required for loading Hugging Face models
]

# Install each package
for package in packages:
    print(f"Installing {package}...")
    install(package)

# Download NLTK data
import nltk
print("Downloading NLTK 'punkt' and 'stopwords' data...")
nltk.download('punkt')
nltk.download('stopwords')

print("All required packages and NLTK data have been installed.")
