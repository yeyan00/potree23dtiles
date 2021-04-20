#!coding=utf-8
from core.potree23dtiles import convert_to_3dtiles
from core.cmd import cmd_exec
import os
import shutil

def las_to_3dtiles(las_file,epsg_str,out_dir):
    file_dir = os.path.dirname(__file__)
    exe_pth = os.path.join(file_dir,'tool/PotreeConverter_2/PotreeConverter.exe')

    tmp_dir = os.path.join(out_dir,'temp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    print('step 1:convertlas start...')
    meta_json_file = os.path.join(tmp_dir,'metadata.json')
    if not os.path.exists(meta_json_file):
        cmd_str = "%s %s -o %s -m random"%(exe_pth,las_file,tmp_dir)
        cmd_exec(cmd_str)

    if os.path.exists(meta_json_file):
        print('step 1:convert las ok')
    else:
        print('step 1:convert las failed')
        return

    print('step 2:convert 3dtiles start...')
    tileset_file =os.path.join(out_dir,'tileset.json')
    if not os.path.exists(tileset_file):
        convert_to_3dtiles(tmp_dir, out_dir, epsg_str)

    if os.path.exists(os.path.join(out_dir,'tileset.json')):
        print('step 2:convert 3dtiles ok')
        shutil.rmtree(tmp_dir)


