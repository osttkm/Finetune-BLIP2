from pathlib import Path
import shutil


file_path = "/home/oshita/vlm/fine-tune_data/mvtec/hazelnut/detail.txt"  # 読み込むテキストファイルのパスを指定

with open(file_path, "r") as file:
    for line in file:
        path = line.split('$')
        import pdb;pdb.set_trace()
        if path.parent.name=='train':
            save_path = '/home/oshita/vlm/fine-tune_data/mvtec/hazelnut/images/train/ok'+path.name
        else:
            save_path = '/home/oshita/vlm/fine-tune_data/mvtec/hazelnut/images/train/'+path.parent.name+path.name
        shutil.copy(str(path), save_path) 
        
        


def get_file_paths(directory):
    directory_path = Path(directory)
    jpg_files = directory_path.glob("*.jpg")  # .jpg拡張子のファイルを取得
    file_paths = [str(file_path) for file_path in jpg_files]
    return file_paths

# 使用例
directory_path = "path/to/directory"  # 対象ディレクトリのパスを指定
jpg_file_paths = get_jpg_file_paths(directory_path)

# 取得したファイルパスを表示
for file_path in jpg_file_paths:
    print(file_path)

path = Path('/home/oshita/vlm/fine-tune_data/mvtec/hazelnut/test/crack')