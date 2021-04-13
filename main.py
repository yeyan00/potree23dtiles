#!coding=utf-8
from core.potree23dtiles import convert23dtiles

if __name__ == "__main__":
    #src-->potree data dir,include cloud.js
    src = r'G:\data\potree_test\potree17'
    # out dir
    outdir = r'G:\data\potree_test\potree17\tiles'
    proj_param = 'EPSG:32649'

    convert23dtiles(src,outdir,proj_param)