import subprocess
import sys
import pkg_resources
import platform

def check_python_version():
    """Check if Python version is compatible (recommended: Python 3.8+)"""
    if sys.version_info < (3, 8):
        raise RuntimeError("This project requires Python 3.8 or higher")
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} detected ✓")

def install_requirements():
    """Install all required packages with specific versions"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "requests>=2.28.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "accelerate>=0.20.0",  # Required for some transformers features
        "scipy>=1.9.0",        # Required for some torch operations
        "tqdm>=4.65.0",        # For progress bars
    ]

    print("Installing required packages...")
    
    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            print(f"CUDA is available (version {cuda_version}) ✓")
        else:
            print("CUDA is not available. Installing CPU-only versions.")
    except ImportError:
        cuda_available = False
        print("Torch not found. Will attempt to install with CUDA support.")

    for requirement in requirements:
        try:
            package_name = requirement.split('>=')[0]
            print(f"Installing {requirement}...")
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                requirement,
                "--upgrade"
            ])
            print(f"Successfully installed {package_name} ✓")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {requirement}: {str(e)}")
            sys.exit(1)

def verify_installations():
    """Verify that all required packages are installed correctly"""
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "PIL": "Pillow",
        "requests": "Requests",
        "transformers": "Transformers",
        "peft": "PEFT",
        "bitsandbytes": "BitsAndBytes",
        "accelerate": "Accelerate"
    }

    print("\nVerifying installations...")
    
    for package, display_name in required_packages.items():
        try:
            if package == "PIL":
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = module.__version__
            print(f"{display_name} version {version} is installed ✓")
        except (ImportError, AttributeError) as e:
            print(f"Warning: {display_name} verification failed: {str(e)}")

def main():
    """Main function to run the installation process"""
    print("Starting installation process...\n")
    
    try:
        check_python_version()
        install_requirements()
        verify_installations()
        
        print("\nAll required packages have been installed successfully! ✓")
        print("You can now run your project.")
        
    except Exception as e:
        print(f"\nAn error occurred during installation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()