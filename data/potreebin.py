#!coding=utf-8
import os
import numpy as np
import json
from collections import OrderedDict


class Node:
    def __init__(self):
        self.byte_offset = 0
        self.byte_size = 0
        self.num_points = 0
        self.hierarchy_byte_offset = 0
        self.hierarchy_byte_size = 0
        self.node_type = 0
        self.name = 'r'
        self.childs=OrderedDict()

def get_numpy_type(str_type):
    if str_type=='int32':
        return np.int32
    elif str_type=='int64':
        return np.int64
    elif str_type=='int16':
        return np.int16
    elif str_type=='uint64':
        return np.uint64
    elif str_type=='uint32':
        return np.uint32
    elif str_type=='uint16':
        return np.uint16
    elif str_type=='uint8':
        return np.uint8
    elif str_type == 'int8':
        return np.int8
    elif str_type=='float':
        return np.float32
    elif str_type=='double':
        return np.float64



class PotreeBin:
    """
    Bin potree2.0 data
    Attributes:

    """
    def __init__(self,path):
        super().__init__()
        self.path = path
        self.nodes={}
        self.metadata = {}
        self.is_open_octree =False
        self.octree_fh = None

    def read_metadata(self):
        """
        get metadata
        :return:
        """
        meta_file = os.path.join(self.path,'metadata.json')

        with open(meta_file) as f:
            self.metadata = json.load(f)

            self.per_point_size = 0
            attrs = self.metadata['attributes']

            dts =[]
            for _attr in attrs:
                self.per_point_size += _attr['size']
                dts.append((_attr['name'], get_numpy_type(_attr['type']), _attr['numElements']))

            self.data_type = np.dtype(dts)



    def check_hierarchy(self):
        if not self.metadata:
            self.read_metadata()

        for k,node in self.nodes.items():
            if node.byte_size/self.per_point_size != node.numPoints:
                return False

        return True


    def read_hierarchy(self):
        """
        read all nodes,save it as dict
        :return:
        """
        if not self.metadata:
            self.read_metadata()
        first_chunk_size = self.metadata['hierarchy']["firstChunkSize"]

        hierarchy_file = os.path.join(self.path, 'hierarchy.bin')

        if not os.path.exists(hierarchy_file):
            print('hierarchy.bin not exists')
            return

        fh = open(hierarchy_file, 'rb')
        buffer = np.fromfile(fh,dtype='u1',count=first_chunk_size)
        bytes_per_node = 22

        def b_node_exists(nodes,node):
            for e in nodes:
                if e and e.name == node.name:return True
            return False

        def parse_node(node,buffer,nodes):
            """
            parse noded,get all nodes,refer to potree->OctreeLoader.js
            :param node:
            :param buffer:
            :param _nodes:
            :return:
            """
            nodes[0] = node
            node_pos = 1
            for i,current in np.ndenumerate(nodes):
                if current == 0:
                    current = Node()
                    current.name = node.name
                _node_type = np.frombuffer(buffer, dtype='u1', offset=i[0] * bytes_per_node, count=1)[0]
                _child_mask = np.frombuffer(buffer, dtype='u1', offset=i[0] * bytes_per_node + 1, count=1)[0]
                current.num_points = np.frombuffer(buffer, dtype='u4', offset=i[0] * bytes_per_node + 2, count=1)[0]
                _byte_offset = np.frombuffer(buffer, dtype='u8', offset=i[0] * bytes_per_node + 6, count=1)[0]
                _byte_size = np.frombuffer(buffer, dtype='u8', offset=i[0] * bytes_per_node + 14, count=1)[0]
                if current.node_type == 2:
                    current.byte_offset = _byte_offset
                    current.byte_size = _byte_size
                    #save it
                    self.nodes[current.name]  = current
                elif _node_type==2:
                    current.hierarchy_byte_offset = _byte_offset
                    current.hierarchy_byte_size = _byte_size

                else:
                    current.byte_offset = _byte_offset
                    current.byte_size = _byte_size

                current.node_type = _node_type

                if current.node_type==2:
                    #if size=0, and the node is itself,contine
                    if current.byte_size==0 and current.name != node.name:
                        fh.seek(current.hierarchy_byte_offset)
                        _buffer = np.fromfile(fh, dtype='u1', count=current.hierarchy_byte_size)
                        _child_num_nodes = int(_buffer.shape[0] / bytes_per_node)
                        if _child_num_nodes==0:
                            continue
                        _child_nodes = np.zeros(_child_num_nodes,dtype = Node)
                        parse_node(current, _buffer,_child_nodes)
                    continue

                for child_index in range(8):
                    child_exists = ((1 << child_index) & _child_mask) != 0
                    if not child_exists: continue

                    child_name = current.name + str(child_index)
                    child = Node()
                    child.name = child_name

                    if not b_node_exists(nodes,child):
                        nodes[node_pos] = child
                        node_pos += 1
                    self.nodes[child.name] = child

        root = Node()
        root.node_type =2
        numNodes = int(buffer.shape[0] / bytes_per_node)
        _nodes = np.zeros(numNodes,dtype = Node)
        parse_node( root,buffer,_nodes)
        fh.close()

    def open_octree(self):
        if not self.metadata:
            self.read_metadata()

        octree_file = os.path.join(self.path, 'octree.bin')

        self.octree_fh = open(octree_file, 'rb')
        self.is_open_octree = True

    def close_octree(self):
        self.octree_fh.close()

    def read_octree_node(self, node):
        if not self.metadata:
            self.read_metadata()
        try:
            if not self.is_open_octree:
                self.open_octree()
            if node and node.byte_size>0:
                self.octree_fh.seek(node.byte_offset)
                data = np.fromfile(self.octree_fh,dtype=self.data_type,count=int(node.byte_size/self.data_type.itemsize))
                return data
        except Exception as err :
            self.close_octree()
            print(err)


    def test_octree_to_txt(self,txt_file,limit_node_num=100):
        """
        read all nodes,cost memory,just for check pointcloud
        :param txt_file:
        :param limitNodes: limit nodes count 2 txt
        :return:
        """
        octree_file = os.path.join(self.path, 'octree.bin')
        if not os.path.exists(octree_file):
            print('octree.bin not exists')
            return

        if not self.metadata:
            self.read_metadata()
        if not self.nodes:
            self.read_hierarchy()

        with open(octree_file, 'rb') as fh:
            count=0
            for k,node in self.nodes.items():
                fh.seek(node.byte_offset)
                data = np.fromfile(fh, dtype=self.data_type, count=int(node.byte_size / self.data_type.itemsize))
                pos = data['position']*self.metadata['scale'][0]
                if count>limit_node_num:break

                #append to file
                with open(txt_file, "ab") as f:
                    np.savetxt(f,pos, delimiter=',')
                count +=1


if __name__ =="__main__":
    # # test dtype
    # dt = np.dtype([('grades', np.float64, 2)])
    # z = np.zeros(2,dtype=dt)
    # x = np.array([[2,3],[4,5]], dtype=dt)

    # 1.read all nodes, then can use potree23tiles convert it
    # pbin.read_hierarchy()
    pth = r'G:\01_code\potree\pointclouds\test'
    pbin = PotreeBin(pth)

    # 2.check read nodes data,export to txt, use cloudcompare open and display
    out_file = r'G:\01_code\potree\pointclouds\test\1.txt'
    if os.path.exists(out_file):os.remove(out_file)
    pbin.test_octree_to_txt(out_file)

    # #3.read one node
    # nodes = pbin.nodes
    # rnode = nodes['r']
    # data = pbin.read_octree_node(rnode)
