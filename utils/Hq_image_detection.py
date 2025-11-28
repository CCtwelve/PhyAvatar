from fileinput import filename
import os
from PIL import Image
from PIL import UnidentifiedImageError
from PIL.ImageQt import rgb
from tqdm import tqdm # Import tqdm for progress bars

# --- 1. Configuration ---

# Your dataset root path
# /mnt/cvda/cvda_phava/dataset/Actor01/Sequence1
# The script will look for '4x/masks' within this path
BASE_DATA_PATH = "/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1"
type='masks'  # masks or rgbs

# Output log file name
if type =='masks':
    filename="mask"
else:
    filename="rgb"

OUTPUT_FILE = f"/mnt/cvda/cvda_phava/dataset/Actor01/Sequence1/4x/corrupted_images_{type}.txt"

# Set loop ranges based on your description
VIEW_START = 1
VIEW_END = 160
FRAME_START = 0
FRAME_END = 2213

# --- 2. Main Script Logic ---

# Construct the absolute path to the masks directory
masks_base_dir = os.path.join(BASE_DATA_PATH, "4x", type)

# Store paths of all corrupted files
bad_files = []

print(f"Starting image check in: {masks_base_dir} ...")
print(f"View range: {VIEW_START} to {VIEW_END}")
print(f"Frame range: {FRAME_START} to {FRAME_END}")
total_files = (VIEW_END - VIEW_START + 1) * (FRAME_END - FRAME_START + 1)
print(f"Total files to check: {total_files} ...")

# Use tqdm to create a progress bar for the outer loop
try:
    view_iterator = tqdm(range(VIEW_START, VIEW_END + 1), desc="Views")
    
    for view_num in view_iterator:
        # Format as three digits (e.g., 001, 002, ..., 160)
        view_index = f"{view_num:03d}"
        
        # Inner loop: frames
        for frame_num in range(FRAME_START, FRAME_END + 1):
            # Format as four digits (e.g., 0000, 0001, ..., 2213)
            fram_index = f"{frame_num:04d}"
            
            # --- Construct File Path ---
            # Based on your template: Cam{view_index}_mask00{fram_index}.png
            # This means the filename is Cam001_mask000000.png, Cam001_mask000001.png ...
            file_name = f"Cam{view_index}_{filename}00{fram_index}.jpg"
            dir_name = f"Cam{view_index}"
            
            # Full absolute path
            image_path = os.path.join(masks_base_dir, dir_name, file_name)
            
            # Relative path format as requested by the user
            relative_path = f"masks/{dir_name}/{file_name}"

            # --- 3. Core Detection Logic ---
            try:
                # Try to open the image
                img = Image.open(image_path)
                
                # Key: img.load() forces reading all image data
                # Your error (OSError: image file is truncated) is triggered at this step
                # (or implicitly called during copy.deepcopy / tobytes)
                img.load()
                
                # If successful, close the file
                img.close()
                
            except (OSError, UnidentifiedImageError) as e:
                # Catch OSError (truncated file) or UnidentifiedImageError (not an image file)
                error_message = f"{relative_path} (Error: {e})"
                print(f"Problem found: {error_message}")
                bad_files.append(error_message)
                
            except FileNotFoundError:
                # Catch FileNotFoundError (file does not exist)
                error_message = f"{relative_path} (Error: File not found)"
                print(f"Problem found: {error_message}")
                bad_files.append(error_message)
                
            except Exception as e:
                # Catch other unknown errors
                error_message = f"{relative_path} (Unknown error: {e})"
                print(f"Problem found: {error_message}")
                bad_files.append(error_message)

except KeyboardInterrupt:
    print("\nDetection interrupted by user.")

# --- 4. Write Results ---
if bad_files:
    print(f"\nCheck complete. Found {len(bad_files)} problematic files.")
    print(f"Writing list to {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, 'w') as f:
        for line in bad_files:
            f.write(line + '\n')
    
    print("Write complete.")
else:
    print("\nCheck complete. No corrupted or missing images found.")