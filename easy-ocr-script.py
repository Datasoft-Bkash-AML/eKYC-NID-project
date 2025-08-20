import easyocr
import re
import os
import platform
from pathlib import Path

def easyocr_method(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    
    for (bbox, text, confidence) in results:
        if confidence > 0.7:
            digits = re.sub(r'[^\d]', '', text)
            if len(digits) == 10:
                return digits
    return None

def detect_environment():
    """
    Detect the current operating system and environment
    """
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    
    print(f"🖥️  Operating System: {system}")
    print(f"📋 Release: {release}")
    print(f"💻 Architecture: {machine}")
    print(f"🐍 Python Version: {platform.python_version()}")
    
    return system

def get_default_path():
    """
    Get default path based on operating system
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows default paths
        return "history\\faf177c7-433d-4fb2-b993-aaea8e0de37a\\FrontSide"
    elif system == "Darwin":  # macOS
        # macOS default paths
        return "history/faf177c7-433d-4fb2-b993-aaea8e0de37a/FrontSide"
    else:  # Linux and other Unix-like systems
        # Linux/Unix default paths
        return "history/faf177c7-433d-4fb2-b993-aaea8e0de37a/FrontSide"

def process_images_cross_platform(base_path=None, num_images=4):
    """
    Process images with automatic OS detection and path handling
    
    Args:
        base_path (str, optional): Base path. If None, uses OS-appropriate default
        num_images (int): Number of images to process
    
    Returns:
        list: List of extracted 10-digit numbers
    """
    # Detect environment
    system = detect_environment()
    print("-" * 50)
    
    # Use default path if none provided
    if base_path is None:
        base_path = get_default_path()
        print(f"📁 Using default path for {system}: {base_path}")
    else:
        print(f"📁 Using provided path: {base_path}")
    
    # Convert to Path object for cross-platform compatibility
    base_path = Path(base_path)
    
    # Generate image paths
    image_paths = [base_path / f"nationalId_TC{i}.jpg" for i in range(num_images)]
    
    results = []
    
    print(f"🔍 Processing {num_images} images...")
    print("-" * 50)
    
    for i, image_path in enumerate(image_paths):
        try:
            # Check if file exists
            if not image_path.exists():
                print(f"  ⚠️  [{i}] File not found: {image_path}")
                continue
            
            print(f"  📷 [{i}] Processing: {image_path.name}")
            result = easyocr_method(str(image_path))
            
            if result:
                print(f"  ✅ [{i}] Found: {result}")
                results.append(result)
            else:
                print(f"  ❌ [{i}] No valid number found")
                
        except Exception as e:
            print(f"  ❌ [{i}] Error: {str(e)}")
            continue
    
    return results

def get_environment_specific_paths():
    """
    Return environment-specific example paths
    """
    system = platform.system()
    
    paths = {
        "Windows": [
            r"C:\Users\YourName\Documents\OCR_Images",
            r"D:\Projects\ImageProcessing",
            r"history\batch1\FrontSide",
        ],
        "Darwin": [  # macOS
            "/Users/YourName/Documents/OCR_Images",
            "/Applications/ImageProcessing",
            "history/batch1/FrontSide",
        ],
        "Linux": [
            "/home/yourname/Documents/OCR_Images",
            "/opt/imageprocessing",
            "history/batch1/FrontSide",
        ]
    }
    
    return paths.get(system, paths["Windows"])

def interactive_cross_platform():
    """
    Interactive mode with OS-aware suggestions
    """
    system = detect_environment()
    print("=" * 50)
    print("Interactive Cross-Platform OCR Processing")
    print("=" * 50)
    
    # Show OS-specific path examples
    example_paths = get_environment_specific_paths()
    print(f"\n📝 Example paths for {system}:")
    for i, path in enumerate(example_paths, 1):
        print(f"  {i}. {path}")
    

# Main execution
if __name__ == "__main__":
    print("🚀 Cross-Platform OCR Processor")
    print("=" * 40)
    
    # Method 1: Automatic detection with default path
    print("\n1️⃣  Using automatic OS detection:")
    results = process_images_cross_platform()
    
    # Method 2: Manual path specification
    print("\n\n2️⃣  Using manual path:")
    # This will work on any OS - Path() handles the conversion
    manual_path = "history/faf177c7-433d-4fb2-b993-aaea8e0de37a/FrontSide"
    results2 = process_images_cross_platform(manual_path)
    
    # Method 3: Interactive mode (uncomment to use)
    print(interactive_cross_platform())
    
    # Environment info summary
    print(f"\n📊 Summary:")
    print(f"   OS: {platform.system()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   Results Method 1: {len(results)} found")
    print(f"   Results Method 2: {len(results2)} found")