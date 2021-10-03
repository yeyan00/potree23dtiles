#!coding=utf-8
import numpy as np
import json

indent = 4

def cfg2dtype(m):
    k0 = set([ (k[:2],v) for k,v in sorted(m.items()) ])
    k1 = set([ (k[2:],v) for k,v in sorted(m.items()) ])
    assert len(k0)==len(m), 'invalid field name!'
    assert len(k1)==len(m), 'invalid field name!'

    return np.dtype([ (k[2:],v) for k,v in sorted(m.items()) ])


class PointCloudV1(object):
    Header = {
        '0-magic': '4S',
        '1-version': 'u4',
        '2-byteLength': 'u4',
        '3-featureTableByteLength': '2u4',  # json,bin
        '4-batchTableByteLength': '2u4',  # json,bin
    }

    FeatureTable = {

    }

    BatchTable = {
        'classification': {"byteOffset":0,"type": "SCALAR","componentType": "UNSIGNED_BYTE"}
    }

class Pnts:
    data = None
    def read(self,filename):
        with open(filename,'rb') as fh:
            hdr = np.memmap(fh, dtype=cfg2dtype(PointCloudV1.Header), offset=0, shape=(1,), mode='r')

        rs = {}
        return rs

    def alignHeader2Bytes(self,data):
        _str = json.JSONEncoder().encode(data)
        if len(_str) % indent != 0:
            for i in range(0, indent - len(_str) % indent):
                _str += ' '
        bytes = np.array(_str.encode('cp936'))
        return bytes

    def write(self,filename,data):
        '''
        :param filename:
        :param data: {feature:None,batch:None}
        :return:  {status:1,msg:''} status=1,successed!
        '''
        rs = dict(status=0,msg='')
        feature_data = data.get('feature')
        batch_data = data.get('batch')

        if (not feature_data) or (not batch_data):
            return rs

        pos = feature_data.get("POSITION")
        rgb = feature_data.get("RGB")

        if pos is None:
            rs['msg'] = "pos data is none"
            return rs

        body_header = {"POINTS_LENGTH": pos.shape[0],
                       "POSITION": {"byteOffset": 0}}

        featureTableSize = pos.nbytes

        if rgb is not None: # check rgb
            body_header['RGB'] = {"byteOffset": pos.nbytes}
            featureTableSize += rgb.nbytes

        body_header_bytes = self.alignHeader2Bytes(body_header)
        batch_header = {}

        hdr = np.zeros((1,), dtype=cfg2dtype(PointCloudV1.Header))

        #generate data
        _data = [hdr, body_header_bytes, pos, rgb]
        _batch_data = []
        byteOffset = 0
        batch_header = PointCloudV1.BatchTable
        for k,v in batch_header.items():
            if batch_data.get(k) is not None:
                batch_header[k]['byteOffset'] = byteOffset
                byteOffset += batch_data[k].nbytes
                _batch_data.append(batch_data[k])

        batch_header_bytes = self.alignHeader2Bytes(PointCloudV1.BatchTable)
        _data.append(batch_header_bytes)
        _data.extend(_batch_data)

        hdr[0]['magic'] = 'pnts'
        hdr[0]['version'] = 1
        hdr[0]['byteLength'] = np.sum(e.nbytes for e in _data)
        hdr[0]['featureTableByteLength'][0] = body_header_bytes.nbytes
        hdr[0]['featureTableByteLength'][1] = pos.nbytes + rgb.nbytes
        hdr[0]['batchTableByteLength'][0] = batch_header_bytes.nbytes
        hdr[0]['batchTableByteLength'][1] = np.sum(e.nbytes for e in _batch_data)

        txt = [e.tobytes() for e in _data]
        txt = b''.join(txt)
        with open(filename,'wb') as fh:
            fh.write(txt)

        rs=dict(status=1,msg='ok')
        return rs

if __name__ == "__main__":
    print(1)