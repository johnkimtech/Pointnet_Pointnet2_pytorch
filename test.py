STL_ROOT = 'data/Foot_Last_stl/STL'
import time

from data_utils import helpers
from pathlib import Path
import re
import os

stl_files_list = [] # input
txt_files_list = [] # output

for root, dirs, files in os.walk(STL_ROOT, topdown=False):
    txt_root = re.sub('(?i)stl', 'txt', root)
    Path(txt_root).mkdir(parents=True, exist_ok=True)
    stl_files_list += [os.path.join(root, file) for file in files]

txt_files_list = list(map(lambda stl_file: re.sub('(?i)stl', 'txt', stl_file), stl_files_list))



import sys
from tqdm import tqdm
import concurrent.futures
from random import randint


def process_files(stl, txt):
    helpers.stl_to_xyz_with_normals_vectorized(stl, txt)
    # converter.stl_to_xyz_with_normals(stl, txt)
    # time.sleep(randint(1,3))


# Create a multithreaded executor
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []

    for stl, txt in zip(stl_files_list, txt_files_list):
        future = executor.submit(process_files, stl, txt)
        futures.append(future)

    # Use tqdm to create a progress bar
    with tqdm(total=len(futures)) as pbar:
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)

# print("All tasks completed.")

# from tqdm import tqdm


# for stl, txt in tqdm(zip(stl_files_list, txt_files_list), total=len(txt_files_list)):
#     # converter.stl_to_xyz_with_normals(stl, txt)
#     converter.stl_to_xyz_with_normals_vectorized(stl, txt)