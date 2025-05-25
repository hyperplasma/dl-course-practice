import os
import shutil

def move_files_from_subdirectories(parent_dir):
    for entry in os.listdir(parent_dir):
        if entry == '.DS_Store':
            continue
        
        subdirectory_path = os.path.join(parent_dir, entry)
        
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                if filename == '.DS_Store':
                    continue
                
                file_path = os.path.join(subdirectory_path, filename)
                
                if os.path.isfile(file_path):
                    shutil.move(file_path, parent_dir)
                    print(f"移动文件: {file_path} 到 {parent_dir}")

            # 删除空的子目录
            if not os.listdir(subdirectory_path):
                os.rmdir(subdirectory_path)
                print(f"删除空目录: {subdirectory_path}")

if __name__ == "__main__":
    target_directory = '/Users/hyperplasma/Pictures/favorite/xlikes'
    move_files_from_subdirectories(target_directory)