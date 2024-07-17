import os
import filecmp

def get_all_files(directory):
    """
    Get all files in the directory, excluding __pycache__.
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), directory))
    return file_list

def get_modified_files(original_dir, modified_dir):
    """
    Compare the files between original and modified directory and get the modified files.
    """
    original_files = get_all_files(original_dir)
    modified_files = get_all_files(modified_dir)

    modified_file_list = []

    for file in modified_files:
        original_file_path = os.path.join(original_dir, file)
        modified_file_path = os.path.join(modified_dir, file)

        if file in original_files:
            if not filecmp.cmp(original_file_path, modified_file_path, shallow=False):
                modified_file_list.append(file)
        else:
            modified_file_list.append(file)

    return modified_file_list

def generate_notice_file(modified_files, output_file='notice.txt'):
    """
    Generate the notice file.
    """
    with open(output_file, 'w') as f:
        f.write("="*80  + "\n")
        f.write("The following files may have been modified by Horizon Robotics Inc. :\n")
        f.write("="*80 + "\n\n")
        for file in modified_files:
            f.write(file + '\n')

def main():
    original_dir = '/home/users/yihan01.hu/workspace/nuplan-devkit'
    modified_dir = '/home/users/yihan01.hu/workspace/GUMP/third_party/nuplan-devkit'
    modified_files = get_modified_files(original_dir, modified_dir)
    generate_notice_file(modified_files)

if __name__ == "__main__":
    main()