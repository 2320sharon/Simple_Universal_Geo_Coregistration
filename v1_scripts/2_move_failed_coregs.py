import pandas as pd
import numpy as np
import os
import shutil

satname  = 'S2'

base_dir = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52"

# coregistered ms directory
coreg_dir = os.path.join(base_dir, 'coregistered')
ms_dir = os.path.join(coreg_dir, satname, 'ms')

# unregistered ms_directory
unreg_ms_dir = os.path.join(base_dir, satname, 'ms')

# if s2 then get the swir directory
if satname == 'S2':
    swir_dir = os.path.join(base_dir, satname, 'swir')

mask_dir = os.path.join(base_dir, satname, 'mask')


# filtered csv file location
csv_path = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime11-04-24__04_30_52\coregistered\S2\ms\transformation_results_filtered.csv"


# 1. read the filtered csv
df = pd.read_csv(csv_path)


# 2. get the filenames that were filtered out (filter_passed == False)
failed_coregs = df[~df['filter_passed']]['filename']
print(f"Failed coregistrations: {len(failed_coregs)}")

# 3. move the files to a new directory called 'failed_coregs' in their present directory
# make a directory called failed_coregs
failed_coreg_dir = os.path.join(coreg_dir, 'failed_coregs')
os.makedirs(failed_coreg_dir, exist_ok=True)

# move the files
for filename in failed_coregs:
    # get the ms file
    ms_file = os.path.join(ms_dir, filename)
    # move the ms file
    if os.path.exists(ms_file) and not os.path.exists(os.path.join(failed_coreg_dir, filename)):
        shutil.move(ms_file, failed_coreg_dir)
        print(f"Moved {filename} to {failed_coreg_dir}")


# 4. Copy the original unregistered files to the coregistered directory
# copy the ms files
for filename in os.listdir(unreg_ms_dir):
    src = os.path.join(unreg_ms_dir, filename)
    dst = os.path.join(ms_dir, filename)
    if os.path.exists(dst):
        continue
    if os.path.exists(src) and os.path.isfile(src):
        shutil.copy(src, dst)
        print(f"Copied {filename} to {dst}")
