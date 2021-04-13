#!coding=utf-8
import numpy as np
from collections import OrderedDict

from lasio import las_
from proj import inv_wgs84, trans_wgs84,wgs84_from,wgs84_trans_matrix
import glob2
from pnts import *

_STEP = 20
geomeotric_space = 16
limit_node_point_size=4

def read_las(fname, attr_list=('rgb','class'),tm='EPSG:32650'):
    '''
    read las file
    :param fname: file name
    :param attr_list:
    :param scale: scale
    :return: return dist
    '''
    las = las_(fname)
    ncout = las.count()

    Step = 1 << _STEP
    boundary = []
    xyz = []

    # if proper not in las,not use it
    records = [key[2:] for key in las._RecordTypes[las.get_record_id()].keys()]

    # init attribute of pointcloud
    attribute = {}
    _tm_attr_list = []
    for k in attr_list:
        if k not in records:
            continue
        attribute[k] = []
        _tm_attr_list.append(k)
    attr_list = _tm_attr_list
    _scale = las.scale

    for i in range(0, ncout, Step):
        arr = las.query(i, i + Step)
        n = arr.shape[0]
        if n == 0:
            continue
        if xyz is not None:
            xyz.append((arr['xyz']).astype('i4'))
        for k in attr_list:
            attribute[k].append(arr[k])
        boundary.append(xyz[len(xyz)-1].min(0)*_scale)
        boundary.append(xyz[len(xyz)-1].max(0)*_scale)

    # if no rgb,set (0,0,0)
    if not attribute.get('rgb',None):
        attribute['rgb']=np.zeros((ncout,3),dtype='u1')

    boundary = np.vstack(boundary)
    boundary = np.vstack([boundary.min(0), boundary.max(0)])
    boundary = (boundary + las.offset).T
    xyz = np.vstack(xyz)

    # save point proper
    for k in attr_list:
        if len(attribute[k][0].shape)>1:
            attribute[k] = np.vstack(attribute[k])
        else:
            attribute[k] = np.hstack(attribute[k])

    attribute['rgb']=attribute['rgb'].astype('u1')
    # convert to dist
    pcd= {
        'xyz': xyz,
        'attr': attribute,
        'metainfo':{
            'box': boundary,
            'scale':las.scale,
            'count':ncout,
            'offset': las.offset,
            'record_id':las.record_id
        }
    }
    #covert_neu(pcd,tm=tm)
    return pcd


def covert_neu(info,tm,transM=None):
    '''
    convert to neu
    :param info:
    :param transM:proj4 param
    :return:no
    '''
    # -- center
    if info is None:
        return None
    _xyz = info.get('xyz')
    _meta = info.get('metainfo')
    _offset = _meta.get('offset')
    _scale = _meta.get('scale')
    if _xyz is not None and _meta is not None:
        if transM is  None:
            mu = _xyz.mean(0) + _offset
            mu = wgs84_from(*(mu), tm=tm)
            popM = trans_wgs84(*mu)  # neu   ->  wgs84
        m = inv_wgs84(transM)  # wgs84 ->  neu
        _xyz = _xyz*_scale+_offset
        _xyz = wgs84_from(*(_xyz.T), tm=tm)
        _xyz = _xyz.dot(m[:3,:3]) + m[3,:3]

        pMax = _xyz.max(0)
        pMin = _xyz.min(0)

        center  = (pMax + pMin)/2
        half = (pMax - pMin) / 2

        bbox = np.r_[center.flatten(), (np.identity(3) * half.max()).flatten()]
        tbbox = np.r_[center.flatten(), (np.identity(3) * half).flatten()]

        #convert to i4,in case the loss of precision
        _xyz = (_xyz/_scale).astype('i4')

        # convert xyz to wgs84 neu
        info['xyz'] = _xyz
        info['neu'] = {
            'bbox':list(bbox.flatten()),
            'scale': _scale,
            'tbbox': list(tbbox.flatten()),
        }
    return info

def pcd2pnts(pcd,outfile):
    xyz = pcd['xyz']
    attr = pcd['attr']
    scale = pcd['neu']['scale']
    xyz = (xyz * scale).astype('f4')
    data = {
        'feature': {
            'POSITION': xyz,
            'RGB': attr['rgb']
        },
        'batch': {
            'class': attr['class']
        },
    }
    feature_data = data.get('feature')
    batch_data = data.get('batch')
    pnts = Pnts()
    pnts.write(outfile, data)


class treeNode:
    def __init__(self):
        self.parent = None
        self.childs = []
        self.key = ''
        self.level = 0
        self.file = ''
        self.hierarchy = 0

    def getParent(self):
        return  self.parent

    def addNode(self,node):
        if node.key == self.key:
            if self.parent:
                self.parent.childs.append(node)
                node.parent = self.parent
                print('gen_tree',node.key)
            return
        elif node.level == self.level +1 and node.key[0:-1] == self.key:
            self.childs.append(node)
            node.parent = self
            print('gen_tree',node.key)
            return
        else:
            for e in self.childs:
                e.addNode(node)

    def setFile(self,file,hierarchyStepSize=5):
        self.file = file
        self.key = os.path.basename(file).split('.')[0]
        self.level = len(self.key) - 1
        if self.level%hierarchyStepSize==0:
            self.hierarchy = self.level/hierarchyStepSize

#after potree,the node exist just one point,this situtation can't make the box,
# so if the number of points less than limit_node_point_size,i just abandon it
def visitNode(childs,tileset_json,tm='',transM=None,outdir=''):
    if not childs:return

    for e in childs:
        _child_node = {
        'boundingVolume': {'box':[]},  # save node box
        'children': [],
        'content': {'url': ''},  #save tightbox , 'boundingVolume': ''
        'geometricError': 0,
        }
        _pcd = read_las(e.file, tm=tm)
        if _pcd['xyz'].shape[0] < limit_node_point_size: continue
        _pcd = covert_neu(_pcd, tm=tm, transM=transM)
        pntsfile = r'%s/%s.pnts' % (outdir,e.key)
        # if e.hierarchy:
        #     #pntsdir = r'%s/%s'%(outdir,e.key[e.hierarchy+1:])
        #     pntsfile = r'%s/%s/%s.pnts' % (outdir,e.key)
        pcd2pnts(_pcd, pntsfile)
        _child_node['boundingVolume']['box'] = _pcd.get('neu').get('bbox')
        _child_node['geometricError'] = _pcd.get('neu').get('bbox')[3] / geomeotric_space
        _child_node['content']['url'] = '%s.pnts' % (e.key)
        tileset_json.append(_child_node)
        print('write node:',e.key)
        if not e.childs:   continue
        visitNode(e.childs,_child_node['children'],tm=tm, transM=transM,outdir=outdir)


import os
import json
def convert23dtiles(src,outdir,proj_param,max_level = 15):
    cloudjs = '%s/cloud.js'%(src)
    with open(cloudjs,'r') as f:
        cloud_data = json.load(f)

    hierarchyStepSize =cloud_data['hierarchyStepSize']

    # get all node
    # las_list = glob2.glob("%s/*.las"%src) #just first hierarchy
    data_dir = "%s/data/r"%(src)
    las_list = glob2.glob("%s/**/*.las" % data_dir)  # convert all

    if not las_list:
        print('can not find las')
        return

    tightBox = cloud_data.get('tightBoundingBox')
    mu = np.array([(tightBox['lx'] + tightBox['ux']) / 2, (tightBox['ly'] + tightBox['uy']) / 2,
                   (tightBox['lz'] + tightBox['uz']) / 2])

    # recode matrix
    mu = wgs84_from(*(mu), tm=proj_param)
    transM = wgs84_trans_matrix(*mu)

    def getkey(name):
        name = os.path.basename(name)
        return len(name)

    # sort as file name length
    las_list.sort(key=getkey)

    root = {
        'boundingVolume': {'box': []},
        'children': [],
        'content': {'url': ''},
        'geometricError': 0,
        'refine': 'ADD',
        'transform': list(transM.flatten()),  # r4x4, neu 2 wgs84
    }

    rootnode = treeNode()

    rootnode.setFile(las_list[0],hierarchyStepSize)
    for e in las_list:
        _node = treeNode()
        _node.setFile(e)
        if _node.level>max_level:
            continue
        rootnode.addNode(_node)

    pcd = read_las(rootnode.file, tm=proj_param)
    pcd = covert_neu(pcd, tm=proj_param, transM=transM)

    pcd2pnts(pcd, r'%s/%s.pnts' % (outdir, rootnode.key))
    root['boundingVolume']['box'] = pcd.get('neu').get('bbox')
    root['geometricError'] = pcd.get('neu').get('bbox')[3] / geomeotric_space
    root['content']['url'] = '%s.pnts' % (rootnode.key)

    visitNode(rootnode.childs, root['children'], proj_param, transM, outdir)
    tileset = {
        'asset': {'version': '0.0'},
        'geometricError': root['geometricError'],
        'root': root,
    }
    json.dump(tileset, open(r'%s/tileset.json' % outdir, 'w'))




if __name__ == "__main__":
    #src-->potree data dir,include cloud.js
    src = r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\potree\pointclouds\test'
    # out dir
    outdir = r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\pcdtest1\potree'
    proj_param = 'EPSG:32649'

    convert23dtiles(src,outdir,proj_param,max_level=5)


