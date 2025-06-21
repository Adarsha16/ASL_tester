import os

main_folder = "MP_Data"

# Loop through all items in the main folder
for name in os.listdir(main_folder):
    subdir_path = os.path.join(main_folder, name)

    # Check if it is a directory and is empty
    if os.path.isdir(subdir_path) and not os.listdir(subdir_path):
        os.rmdir(subdir_path)
        print(f"Deleted empty directory: {subdir_path}")
