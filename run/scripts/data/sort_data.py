import os
import argparse
from glob import glob

def rename_files(data_dir):
    # Loop through modes: train, val, test
    for mode in ['train', 'val', 'test']:
        print(f'Renaming {mode}', flush=True)
        mode_dir = os.path.join(data_dir, mode)
        
        # Find all the files with the given prefixes
        files = glob(os.path.join(mode_dir, '[XyUpI]_*.pt'))
        
        # Dictionary to store unique uuids and assign them an integer i
        uuid_map = {}
        i = 0
        
        for file in files:
            # Extract the prefix and uuid
            base_name = os.path.basename(file)
            prefix, uuid = base_name.split('_')
            uuid = uuid.split('.pt')[0]

            # If we haven't seen this uuid before, assign it an index i
            if uuid not in uuid_map:
                uuid_map[uuid] = i
                i += 1

            # Get the new filename with the index i
            new_filename = f"{prefix}_{uuid_map[uuid]}.pt"
            new_file_path = os.path.join(mode_dir, new_filename)

            # Rename the file and delete the old one
            os.rename(file, new_file_path)
            print(f"Renamed {file} to {new_file_path}", flush=True)

if __name__ == "__main__":
    # Argument parser to get the data_dir from the shell argument
    parser = argparse.ArgumentParser(description="Rename and replace dataset files")
    parser.add_argument("--data_dir", type=str, help="The path to the data directory")
    
    args = parser.parse_args()
    
    # Call the rename function
    rename_files(args.data_dir)

