#!coding=utf-8
from data.lasio import las_
from data.potreebin import PotreeBin
from core.proj import inv_wgs84, trans_wgs84,wgs84_from,wgs84_trans_matrix
import glob2
from data.pnts import Pnts
import os
import json
import numpy as np


_STEP = 20
geomeotric_space = 16
limit_node_point_size=4

def read_las(fname, attr_list=('rgb','classification')):
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
    return pcd

def read_bin(potree,node,attr_list=('rgb','classification')):
    data = potree.read_octree_node(node)
    #data to pcd
    xyz = data['position']
    scale = potree.metadata['scale'][0]
    offset = potree.metadata['offset']
    attribute={}

    for i,e in enumerate(potree.metadata['attributes']):
        if e['name'] in attr_list:
                attribute[e['name']] = data[e['name']]

    # must include rgb
    if attribute.get('rgb',None) is None:
        attribute['rgb']=np.zeros((xyz.shape[0],3),dtype='u1')
    else:
        if attribute.get('rgb', None).max()>255:
            attribute['rgb'] = (attribute.get('rgb',None)/255 +0.5).astype('u1')
        else:
            attribute['rgb'] = attribute.get('rgb', None).astype('u1')

    pcd = {
        'xyz': xyz,
        'attr': attribute,
        'metainfo': {
            'scale': scale,
            'offset': offset
        }
    }
    return pcd

def covert_ecef(info, tm, trans_mat=None):
    '''
    convert to ecef as wgs84
    :param info:
    :param trans_mat:proj4 param
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
        m = inv_wgs84(trans_mat)  # wgs84 ->  neu
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

def pcd2_pnts(pcd, outfile):
    xyz = pcd['xyz']
    attr = pcd['attr']
    scale = pcd['neu']['scale']
    xyz = (xyz * scale).astype('f4')

    #just use position,rgb,classification for test
    data = {
        'feature': {
            'POSITION': xyz,
            'RGB': attr['rgb']
        },
        'batch': {
            'classification': attr['classification']
        },
    }
    # feature_data = data.get('feature')
    # batch_data = data.get('batch')
    pnts = Pnts()
    pnts.write(outfile, data)


class TreeNode:
    def __init__(self):
        self.parent = None
        self.childs = []
        self.key = ''
        self.level = 0
        self.file = ''

        # for Node 2
        self.byte_offset = 0
        self.byte_size = 0
        #self.hierarchy = 0

    def get_parent(self):
        return  self.parent

    def add_node(self, node):
        if node.key == self.key:
            if self.parent:
                self.parent.childs.append(node)
                node.parent = self.parent
                #print('gen_tree',node.key)
            return
        elif node.level == self.level +1 and node.key[0:-1] == self.key:
            self.childs.append(node)
            node.parent = self
            #print('gen_tree',node.key)
            return
        else:
            for e in self.childs:
                e.add_node(node)

    def set_file(self,file,hierarchy_step_size=5):
        self.file = file
        self.key = os.path.basename(file).split('.')[0]
        self.level = len(self.key) - 1
        # if self.level%hierarchy_step_size==0:
        #     self.hierarchy = self.level/hierarchy_step_size

    def set_node(self,node):
        self.key = node.name
        self.level = len(self.key) - 1

        self.byte_offset = node.byte_offset
        self.byte_size = node.byte_size


#after potree,the node exist just one point,this situtation can't make the box,
# so if the number of points less than limit_node_point_size,i just abandon it
def visit_node(childs, tileset_json, tm='', trans_mat=None, outdir='',potree_object = None):
    if not childs:return

    for e in childs:
        _child_node = {
        'boundingVolume': {'box':[]},  # save node box
        'children': [],
        'content': {'url': ''},  #save tightbox , 'boundingVolume': ''
        'geometricError': 0,
        }

        if e.byte_size>0 and potree_object is not None:
            _pcd = read_bin(potree_object,e)
        else:
            _pcd = read_las(e.file)
        if _pcd['xyz'].shape[0] < limit_node_point_size: continue
        _pcd = covert_ecef(_pcd, tm=tm, trans_mat=trans_mat)
        pntsfile = r'%s/%s.pnts' % (outdir,e.key)
        # if e.hierarchy:
        #     #pntsdir = r'%s/%s'%(outdir,e.key[e.hierarchy+1:])
        #     pntsfile = r'%s/%s/%s.pnts' % (outdir,e.key)
        pcd2_pnts(_pcd, pntsfile)
        _child_node['boundingVolume']['box'] = _pcd.get('neu').get('bbox')
        _child_node['geometricError'] = _pcd.get('neu').get('bbox')[3] / geomeotric_space
        _child_node['content']['url'] = '%s.pnts' % (e.key)
        tileset_json.append(_child_node)
        #print('write node:',e.key)
        if not e.childs:   continue
        visit_node(e.childs, _child_node['children'], tm=tm, trans_mat=trans_mat, outdir=outdir,potree_object=potree_object)



def convert_to_3dtiles_v1(src,outdir,proj_param,max_level = 15):
    """
    for potreeconvert version before 2,test 1.7
    :param src:
    :param outdir:
    :param proj_param:
    :param max_level:
    :return:
    """
    cloudjs = '%s/cloud.js'%(src)
    with open(cloudjs,'r') as f:
        cloud_data = json.load(f)

    hierarchy_step_size =cloud_data['hierarchyStepSize']

    # get all node
    # las_list = glob2.glob("%s/*.las"%src) #just first hierarchy
    data_dir = "%s/data/r"%(src)
    las_list = glob2.glob("%s/**/*.las" % data_dir)  # convert all

    if not las_list:
        print('can not find las')
        return

    tight_box = cloud_data.get('tightBoundingBox')
    mu = np.array([(tight_box['lx'] + tight_box['ux']) / 2, (tight_box['ly'] + tight_box['uy']) / 2,
                   (tight_box['lz'] + tight_box['uz']) / 2])

    # recode matrix
    mu = wgs84_from(*(mu), tm=proj_param)
    trans_mat = wgs84_trans_matrix(*mu)

    def get_key(name):
        name = os.path.basename(name)
        return len(name)

    # sort as file name length
    las_list.sort(key=get_key)

    root = {
        'boundingVolume': {'box': []},
        'children': [],
        'content': {'url': ''},
        'geometricError': 0,
        'refine': 'ADD',
        'transform': list(trans_mat.flatten()),  # r4x4, neu 2 wgs84
    }

    rootnode = TreeNode()

    rootnode.set_file(las_list[0],hierarchy_step_size)
    for e in las_list:
        _node = TreeNode()
        _node.set_file(e)
        if _node.level>max_level:
            continue
        rootnode.add_node(_node)

    pcd = read_las(rootnode.file)
    pcd = covert_ecef(pcd, tm=proj_param, trans_mat=trans_mat)

    pcd2_pnts(pcd, r'%s/%s.pnts' % (outdir, rootnode.key))
    root['boundingVolume']['box'] = pcd.get('neu').get('bbox')
    root['geometricError'] = pcd.get('neu').get('bbox')[3] / geomeotric_space
    root['content']['url'] = '%s.pnts' % (rootnode.key)

    visit_node(rootnode.childs, root['children'], proj_param, trans_mat, outdir)
    tileset = {
        'asset': {'version': '0.0'},
        'geometricError': root['geometricError'],
        'root': root,
    }
    json.dump(tileset, open(r'%s/tileset.json' % outdir, 'w'))

def convert_to_3dtiles_v2(src,outdir,proj_param,max_level = 15):
    """
    for potreeconvert version after 2,test 2.0
    :param src:
    :param outdir:
    :param proj_param:
    :param max_level:
    :return:
    """
    potree_bin = PotreeBin(src)
    potree_bin.read_hierarchy()

    attributes = potree_bin.metadata.get('attributes')

    mu = None
    for e in attributes:
        if e['name'] == "position":
            mu = np.array([(e['min'][0]+ e['max'][0]) / 2, (e['min'][1]+ e['max'][1]) / 2,
                           (e['min'][2]+ e['max'][2]) / 2])

    # recode matrix
    mu = wgs84_from(*(mu), tm=proj_param)
    trans_mat = wgs84_trans_matrix(*mu)

    node_list = list(potree_bin.nodes.keys())

    def get_key(name):
        return len(name)

    # sort as file name length
    node_list.sort(key=get_key)

    root = {
        'boundingVolume': {'box': []},
        'children': [],
        'content': {'url': ''},
        'geometricError': 0,
        'refine': 'ADD',
        'transform': list(trans_mat.flatten()),  # r4x4, neu 2 wgs84
    }

    rootnode = TreeNode()

    rootnode.set_node(potree_bin.nodes[node_list[0]])
    #rootnode.set_file(node_list[0], hierarchy_step_size)
    for n in node_list:
        _node = TreeNode()
        _node.set_node(potree_bin.nodes[n])
        if _node.level > max_level:
            continue
        rootnode.add_node(_node)

    pcd = read_bin(potree_bin,rootnode)
    pcd = covert_ecef(pcd, tm=proj_param, trans_mat=trans_mat)

    pcd2_pnts(pcd, r'%s/%s.pnts' % (outdir, rootnode.key))
    root['boundingVolume']['box'] = pcd.get('neu').get('bbox')
    root['geometricError'] = pcd.get('neu').get('bbox')[3] / geomeotric_space
    root['content']['url'] = '%s.pnts' % (rootnode.key)

    visit_node(rootnode.childs, root['children'], proj_param, trans_mat, outdir,potree_object=potree_bin)

    tileset = {
        'asset': {'version': '0.0'},
        'geometricError': root['geometricError'],
        'root': root,
    }
    json.dump(tileset, open(r'%s/tileset.json' % outdir, 'w'))


def convert_to_3dtiles(src,outdir,proj_param,max_level = 15):

    cloudjs = '%s/cloud.js' % (src)
    meta_json = '%s/metadata.json' % (src)

    # check version by file
    if os.path.exists(cloudjs) and os.path.exists('%s/sources.json' % (src)) :
        convert_to_3dtiles_v1(src,outdir,proj_param,max_level)
    elif os.path.exists(meta_json) and os.path.exists('%s/octree.bin' % (src)) and os.path.exists('%s/hierarchy.bin' % (src)):
        convert_to_3dtiles_v2(src,outdir,proj_param,max_level)
    else:
        print('not support!!!')



if __name__ == "__main__":
    # f = r'G:\data\potree_test\potree17\data\r\r.las'
    # p = read_las(f)
    pass


