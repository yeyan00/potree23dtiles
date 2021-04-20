#!coding=utf-8
if __name__ == "__main__":

    from core.potree23dtiles import convert_to_3dtiles

    proj_param = 'EPSG:32649'

    def test_convert_v1():
        # 1.test potreeconvert 1.7
        #src-->potree data dir,include cloud.js
        src = r'G:\data\potree_test\potree17'
        # out dir
        outdir = r'G:\data\potree_test\potree17\tiles'
        convert_to_3dtiles(src,outdir,proj_param)

    def test_convert_v2():
        # 2. test potreeconvert 2.0
        src = r'G:\data\potree_test\potree2'
        # out dir
        outdir = r'G:\data\potree_test\potree2\tiles'
        convert_to_3dtiles(src, outdir, proj_param)

    def test_las_to_3dtiles():
        from core.las23dtiles import las_to_3dtiles
        src = r'G:\data\potree_test\test.las'
        # out dir
        outdir = r'G:\data\potree_test\test'

        las_to_3dtiles(src,proj_param,outdir)

    test_convert_v1()