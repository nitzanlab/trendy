import os
import argparse

# Define the function that performs the file deletion based on limits
def delete_files(data_dir):
    modes = ["train", "val", "test"]
    limits = {"train": 2499, "val": 599, "test": 299}

    # Loop through each mode and delete files based on index
    for mode in modes:
        mode_dir = os.path.join(data_dir, mode)

        if not os.path.exists(mode_dir):
            print(f"Directory {mode_dir} does not exist.")
            continue

        # Get the limit for this mode
        limit = limits[mode]

        # Loop through files in the directory
        for filename in os.listdir(mode_dir):
            # Extract the index <i> from the filename <prefix>_<i>.pt
            try:
                index = int(filename.split('_')[-1].split('.')[0])

                # Delete files with an index greater than the limit
                if index > limit:
                    file_path = os.path.join(mode_dir, filename)
                    os.remove(file_path)
                    print(f"Deleted {file_path}")

            except ValueError:
                print(f"Skipping {filename}, does not match expected format.")

# Set up the argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete files based on index in different directories.")
    parser.add_argument('--data_dir', type=str, help="The base directory containing the train/val/test subdirectories.")
    
    args = parser.parse_args()

    # Call the function with the provided base directory
    delete_files(args.data_dir)
