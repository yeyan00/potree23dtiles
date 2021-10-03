#!coding=utf-8

import logging
import os, time
import numpy


class _BaseErr(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class WarningErr(_BaseErr):
    pass


def _profile(s, msg=''):
    cur = time.time()
    logging.debug('%s:\t%.3f' % (msg, cur - s))
    return cur


def cfg2dtype(m):
    k0 = set([(k[:2], v) for k, v in sorted(m.items())])
    k1 = set([(k[2:], v) for k, v in sorted(m.items())])
    assert len(k0) == len(m), 'invalid field name!'
    assert len(k1) == len(m), 'invalid field name!'

    return numpy.dtype([(k[2:], v) for k, v in sorted(m.items())])


def cfg2ttype(m):
    import tables
    return dict([(k[2:], tables.Col.from_dtype(numpy.dtype(v))) for k, v in sorted(m.items())])


class LasVersion1V2(object):
    Header = {
        '0-filesignature': '4S',
        '1-filesourceid': 'u2',
        '2-global_encoding': 'u2',
        '3-guid': '16b',

        '4-version': '2b',
        '5-systemid': '32S',
        '6-softwareid': '32S',

        '7-file_day': 'u2',
        '8-file_year': 'u2',
        '9-headersize': 'u2',
        'a-offsetdata': 'u4',
        'b-len_vlrs': 'u4',

        'c-dataformatid': '1b',
        'd-sizeofrecord': 'u2',

        'e-numberofdata': 'u4',
        'f-return_points': '5u4',

        'g-scale': '3f8',
        'h-offset': '3f8',
        'i-boundary': '(3,2)f8',
    }

    Record_0 = {
        '0-xyz': '3i4',
        '1-intensity': 'u2',
        '2-return_info': 'b',  # return_number_mask:7
        '3-classification': 'u1',
        '4-angle': 'b',
        '5-user': 'b',
        '6-psid': 'u2',
    }

    Record_1 = {
        '0-xyz': '3i4',
        '1-intensity': 'u2',
        '2-return_info': 'b',  # return_number_mask:7
        '3-classification': 'u1',
        '4-angle': 'b',
        '5-user': 'b',
        '6-psid': 'u2',
        '7-time_stamp': 'f8',
    }
    Record_2 = {
        '0-xyz': '3i4',
        '1-intensity': 'u2',
        '2-return_info': 'b',  # return_number_mask:7
        '3-classification': 'u1',
        '4-angle': 'b',
        '5-user': 'b',
        '6-psid': 'u2',
        # '7-time_stamp':   'f8',
        '8-rgb': '3u2',

    }
    Record_3 = {
        '0-xyz': '3i4',
        '1-intensity': 'u2',
        '2-return_info': 'b',  # return_number_mask:7
        '3-classification': 'u1',
        '4-angle': 'b',
        '5-user': 'b',
        '6-psid': 'u2',
        '7-time_stamp': 'f8',
        '8-rgb': '3u2',
    }
    Record_4 = {
        '0-xyz': '3i4',
        '1-intensity': 'u2',
        '2-return_info': 'b',
        '3-classification': 'u1',
        '4-angle': 'b',
        '5-user': 'b',
        '6-psid': 'u2',
        '7-time_stamp': 'f8',
        '8-wave_packet_index': 'b',
        '9-wave_offset': 'u8',
        'a-wave_packet_size': 'u4',
        'b-wave_return_point': 'f4',
        'c-xyz_t': '3f4',
    }
    Record_5 = {
        '0-xyz': '3i4',
        '1-intensity': 'u2',
        '2-return_info': 'b',
        '3-classification': 'u1',
        '4-angle': 'b',
        '5-user': 'b',
        '6-psid': 'u2',
        '7-time_stamp': 'f8',
        '8-rgb': '3u2',
        '9-wave_packet_index': 'b',
        'a-wave_offset': 'u8',
        'b-wave_packet_size': 'u4',
        'c-wave_return_point': 'f4',
        'd-xyz_t': '3f4',
    }

    VLR = {
    }


class LasVersion2V0(object):
    Header = {
        '0-filesignature': '4S',  # LASF
        '1-filesourceid': 'u4',
        '2-guid1': 'u4',

        '3-guid2': 'u2',
        '4-guid3': 'u2',

        '5-guid4': 'u8',
        '6-version': '2b',  # 2,0

        '7-systemid': '32S',  # GEO-3D_SYSTEM0000
        '8-softwareid': '32S',  # T3D_Capture0000

        '9-file_day': 'u4',
        'a-file_time': 'u4',

        'b-headersize': 'u2',  # 322
        'c-sourceid': 'u2',  # 0
        'd-offsetmeta': 'u2',  # 322
        'e-sizeofmeta': 'u2',  # 337
        'f-sizeofrecord': 'u2',  # 92

        'g-offsetdata': 'u8',
        'h-numberofdata': 'u8',

        'i-offsetvlr': 'u4',  # 322+337=659
        'j-numberofvlr': 'u4',  # 4

        'k-numberofreturn': '16u4',  # [numberofdata,0,0,0]
        'l-compatibility': 'b',  # 255
        'm-cstype': 'b',  # 1
        'n-units': '2b',  # [ 1, 1 ]

        'o-origin': '3f8',  # (0,0,0)
        'p-applyscaling': 'b',  #
        'q-scale': '3f8',
        'r-applyoffset': 'b',
        's-offset': '3f8',
        't-boundary': '(3,2)f8',
    }

    Record = {
        '0-xyz': '3f8',
        '1-intensity': 'f4',
        '2-classification': 'u1',
        '3-psid': 'u2',
        '4-attr': 'b',
        '5-lateralrange': 'f4',
        '6-rgb': '3b',
        '7-confidence': 'b',
        '8-angle': '2f4',  # H,V
        '9-distance': 'f4',
        'a-stddev': 'f4',
        'b-scansize': 'u4',
        'c-time_stamp': 'f8',
        'd-tt': 'f8',
        'f-delta': '2f8',  # [PPS, Sync]
    }

    VlrHeader = {
        '0-uid': '16S',
        '1-rid': 'u2',
        '2-desc': '32S',
        '3-length': 'u8',
    }

    VlrData2 = {
        '0-version': '2u4',
        '1-cmp_type': 'u4',
        '2-cmp_desc': '32S',
        '3-comment': '512S',  # TCHAR
        '4-sn': '128S',
        '5-model': '128S',
    }

    VlrData3 = {
        '0-gps_header': 'f8',
        '1-gps_dxyz': '3f8',
        '2-gps_support': '3f8',
        '3-rp': '2f8',  # roll - pitch
        '4-height': 'f8',
        '5-title_size': 'i4',
        '6-rp_content': '2i2',
        '7-cam_rotation': 'i2',
        '8-stereo_model': 'i4',
        '9-cam_id': '32S',
    }


# -----------------------------------
class QuickTerrain(object):
    Marker = {
        '0-_a': ('b', 1048),  # 0
        '1-xyz': ('f8', 3),  # 1048
        '2-_b': ('b', 56 + 5),  # 1072
        '3-_name': ('S', 10),  # 1133
        '4-_c': ('b', 4 + 4),  # 1143
        '5-name': ('S', 10),  # 1151
        '6-_d': ('b', 952),  # 1161

    }


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

    }


def las_(fname, version=1.2):
    assert os.path.dirname(fname)
    if version == 1.2:
        return LasFile1V2(fname).read()
    raise WarningErr('invalid version %s!' % version)


# -- iterator ----------------------------------------------------------
class BaseRcdsItr(object):
    def __getitem__(self, k):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError


class _LasRcdsIterator(object):  # using for access _LasFile records
    def __init__(self, las):
        self._las = las
        self.shape = (las.count(),)
        self.dtype = las.record_type(las.record_id)

    def __getitem__(self, k):
        if isinstance(k, slice):
            assert k.step is None, 'only support continue access'
            return self._las.query(k.start, k.stop)

        if isinstance(k, int):
            return self._las.query(k, k + 1)

        if isinstance(k, str):
            return self._las.query(None, None)[k]

        raise TypeError("Invalid argument type.")


class _Las1V2_RcdXyz_Saving_Iterator(object):  # using for saving record-xyz( scaled )
    def __init__(self, rcdxyz, rcdtype, fields={}):
        self._arr = rcdxyz
        self._fields = fields
        self._buf = numpy.zeros((1 << 20,), dtype=rcdtype)
        self.shape = (self._arr.shape[0],)  # interface:     shape
        self.dtype = rcdtype  # interface:     dtype

    def __getitem__(self, k):  # interface:     []
        rcdxyz = self._arr[k]
        N = rcdxyz.shape[0]
        self._buf[0:N]['xyz'] = rcdxyz
        for kname, v in self._fields.items():
            self._buf[0:N][kname] = v[k]
        return self._buf[0:N]


class _Las1V2_Xyz_Saving_Iterator(object):  # using for saving xyz( real value, no scaled )
    def __init__(self, xyz, rcdtype, scale, offset, fields={}):
        self._arr = xyz
        self._scale = scale
        self._offset = offset
        self._fields = fields
        self._buf = numpy.zeros((1 << 20,), dtype=rcdtype)
        self.shape = (self._arr.shape[0],)  # interface:     shape
        self.dtype = rcdtype  # interface:     dtype

    def __getitem__(self, k):  # interface:     []
        xyz = self._arr[k]
        N = xyz.shape[0]
        self._buf[0:N]['xyz'] = ((xyz - self._offset) / self._scale).astype('i4')
        for kname, v in self._fields.items():
            self._buf[0:N][kname] = v[k]
        return self._buf[0:N]


# -- core --------------------------------------------------------------
class _LasHeader(object):
    _RecordTypes = {
        0: LasVersion1V2.Record_0,
        1: LasVersion1V2.Record_1,
        2: LasVersion1V2.Record_2,
        3: LasVersion1V2.Record_3,
    }
    recordSizes = [20, 28, 26, 34]
    _MaxLimit = 1 << 32

    def __init__(self, hdr, vlrs):
        self._hdr = hdr
        self._vlrs = None
        self.set_vlrs(vlrs)
        self._points_count = None
        self._my_rcdtype = None

    ##
    def get_offsetdata(self):
        return self._hdr['offsetdata']

    def set_offsetdata(self, arr):
        self._hdr['offsetdata'] = arr

    offsetdata = property(get_offsetdata, set_offsetdata)

    ##
    def get_version(self):
        return self._hdr['version']

    def set_version(self, arr):
        self._hdr['version'][:] = arr

    version = property(get_version, set_version)

    ##
    def get_scale(self):
        return self._hdr['scale']

    def set_scale(self, arr):
        self._hdr['scale'][:] = arr

    scale = property(get_scale, set_scale)

    def get_offset(self):
        return self._hdr['offset']

    def set_offset(self, arr):
        self._hdr['offset'][:] = arr

    offset = property(get_offset, set_offset)

    ##
    def get_header_size(self):
        return self._hdr['headersize']

    def set_header_size(self, arr):
        self._hdr['headersize'] = arr

    header_size = property(get_header_size, set_header_size)

    ##
    def get_vlrs(self):
        return (self._hdr['len_vlrs'], self._vlrs)

    def set_vlrs(self, nv):
        assert isinstance(nv, tuple) and len(nv) == 2
        n, v = nv
        if v is None:
            return
        self._vlrs = v
        self._hdr['len_vlrs'] = n
        self._hdr['offsetdata'] = self._hdr['headersize'] + len(v)

    vlrs = property(get_vlrs, set_vlrs)

    def get_header(self):
        return None if self._hdr is None else self._hdr.copy()

    header = property(get_header, )

    def count(self):
        return self._points_count

    def get_record_id(self):
        return self._hdr['dataformatid']

    def set_record_id(self, v):
        assert v in self._RecordTypes.keys(), 'Record ID is not valid. Unexpected value of %s' % v
        self._hdr['dataformatid'] = v

    record_id = property(get_record_id, set_record_id)

    def clone_header(self, src):
        self.set_version(src.version)
        self.set_header_size(src.header_size)
        self.set_scale(src.scale)
        self.set_offset(src.offset)
        self.set_record_id(src.record_id)
        self.set_vlrs(src.vlrs)
        self._hdr['global_encoding'] = src.header['global_encoding']
        self._hdr['numberofdata'] = 0
        self._hdr['return_points'][:] = 0
        self._hdr['boundary'][:] = 0
        self._hdr['sizeofrecord'] = self.recordSizes[self.get_record_id()]

    def record_type(self, rcdid):
        return cfg2dtype(self._RecordTypes[rcdid])

    def header_type(self):
        raise NotImplementedError


class _LasFile(_LasHeader):
    def __init__(self, hdr, vlrs, fname):
        _LasHeader.__init__(self, hdr, vlrs)
        self.fname = fname
        self._src = None

    def read(self):
        assert self.fname, '_LasFile.fname should not be null!'
        self._src = open(self.fname, mode='rb')
        self._hdr = numpy.fromfile(self._src, dtype=self.header_type(), count=1)[0]
        self._validate_header()
        self._read_more()
        # Checking actual points count
        self._src.seek(0, 2)

        _size = (self._src.tell() - self._hdr['offsetdata']) // self._hdr['sizeofrecord']
        if _size != self._hdr['numberofdata']:
            logging.warning('actual points count is %d, but header says %s' % (_size, self._hdr['numberofdata'],))
        self._points_count = _size

        # Checking for record size
        if self._hdr['sizeofrecord'] != self.recordSizes[self.record_id]:
            assert self._hdr['sizeofrecord'] > self.recordSizes[self.record_id], 'miss some fields'
            logging.warning('data type implies a record size of %s, but header says %d' % (
            self.recordSizes[self.record_id], self._hdr['sizeofrecord']))
            logging.warning('trying to change data type to avoid loss of data...')

            # -- add other fileds
            self._my_rcdtype = self._RecordTypes[self.record_id]
            self._my_rcdtype['_zero'] = '%su1' % (self._hdr['sizeofrecord'] - self.recordSizes[self.record_id])
            self._my_rcdtype = cfg2dtype(self._my_rcdtype)

            # try:
            #    self.set_record_id(self.recordSizes.index(self._hdr['sizeofrecord']))
            # except ValueError:
            #    logging.warning('no data type matches the header record size, data loss may occur...')

        return self

    def query(self, a=None, b=None):
        assert self._hdr is not None, '_LasFile._hdr should not be None!'
        assert self._src is not None, '_LasFile._src should not be None!'

        a, b, _ = slice(a, b, 1).indices(self.count())
        at = self._hdr['offsetdata'] + numpy.int64(a) * (self._hdr['sizeofrecord'])
        self._src.seek(int(at), 0)
        return numpy.fromfile(
            self._src,
            dtype=self._my_rcdtype or self.record_type(self.record_id),
            count=int(b - a)
        )

    def records(self):
        return _LasRcdsIterator(self)

    def close(self):
        if self._src:   self._src.close()

    def memmap(self, mode='r'):  # can be use to modify the las file
        assert self.fname, '_LasFile.fname should not be null!'

        return numpy.memmap(
            self.fname,
            mode=mode,
            offset=int(self._hdr['offsetdata']),
            dtype=self.record_type(self.record_id),
            shape=self.count()
        )

    def save_to(self, rcds, fname,
                keepVLR=True):  # base self config,save to another las, fix it later!!! when rcds.dtype!=self.dtype
        fh, _buf = self._save_begin(rcds.dtype, fname, keepVLR=keepVLR)
        assert fh is not None, 'fh should not be None'

        _STEP = 1 << 20
        s = time.time()
        for i in range(0, rcds.shape[0], _STEP):
            arr = rcds[i:i + _STEP]
            if arr is None or arr.shape[0] == 0:  continue

            arr.tofile(fh)
            fh.flush()

            _buf['mx'].append(arr['xyz'].max(0))
            _buf['mn'].append(arr['xyz'].min(0))

            if (_buf['hdr']['numberofdata'] + arr.shape[0]) > self._MaxLimit:
                logging.warning('las file only suport points less then 4294,967,296!')
            _buf['hdr']['numberofdata'] += arr.shape[0]
            s = _profile(s, 'save to i|%s  size|%s' % (i, arr.shape[0]))

        self._save_end(fh, _buf)

    def make_vlrs(self, name):
        raise NotImplementedError

    def make_header(self, rcdid):
        raise NotImplementedError

    def _validate_header(self):
        raise NotImplementedError

    def _read_more(self):
        raise NotImplementedError

    # ----------------------------
    def _save_begin(self, rcdtype, fname, keepVLR=True):
        hdr = self.header
        if keepVLR:
            (nvlrs, vlrs) = self.vlrs
        else:
            (nvlrs, vlrs) = (0, None)

        # Added an extra line for no rgb and no time stamp (las 1.1)
        # Added two extra lines for las 1.3
        if 'wave_packet_index' in rcdtype.names and 'rgb' in rcdtype.names:
            hdr['dataformatid'] = 5
        elif 'wave_packet_index' in rcdtype.names and not 'rgb' in rcdtype.names:
            hdr['dataformatid'] = 4
        elif 'rgb' in rcdtype.names and 'time_stamp' in rcdtype.names:
            hdr['dataformatid'] = 3
        elif 'rgb' in rcdtype.names and not 'time_stamp' in rcdtype.names:
            hdr['dataformatid'] = 2
        elif not 'rgb' in rcdtype.names and 'time_stamp' in rcdtype.names:
            hdr['dataformatid'] = 1
        else:
            hdr['dataformatid'] = 0

        hdr['numberofdata'] = 0
        hdr['return_points'][:] = 0
        hdr['boundary'][:] = 0

        hdr['len_vlrs'] = nvlrs
        if vlrs is None:
            hdr['offsetdata'] = hdr['headersize']
        else:
            hdr['offsetdata'] = hdr['headersize'] + vlrs.size

        # if not force:
        #    assert not os.path.exists(fname), 'File(%s) will be overwrite! if this is ture, please set force=True!'

        fh = open(fname, 'wb')
        hdr.tofile(fh)
        if not vlrs is None:        vlrs.tofile(fh)

        return fh, {
            'mx': [],
            'mn': [],
            'hdr': hdr,
        }

    def _save_end(self, fh, buf):
        hdr = buf['hdr']
        hdr['boundary'][:, 0] = numpy.array(buf['mx']).max(0).astype('f8') * hdr['scale'] + hdr['offset']
        hdr['boundary'][:, 1] = numpy.array(buf['mn']).min(0).astype('f8') * hdr['scale'] + hdr['offset']
        fh.seek(0)
        hdr.tofile(fh)
        fh.close()


# ----------------------------------------------------
class LasFile1V2(_LasFile):
    def __init__(self, fname=None, rcdid=1):
        _LasFile.__init__(self, self.make_header(rcdid=rcdid), (0, None), fname)

    def rcdxyz_saving_itr(self, arr, rcdid, **kwds):
        rcdtype = self.record_type(rcdid)
        kwds = self._check_fields(rcdtype, kwds)
        return _Las1V2_RcdXyz_Saving_Iterator(arr, rcdtype, fields=kwds)

    def xyz_saving_itr(self, arr, rcdid, **kwds):  # rgb=None, cls=None, intensity=None):
        rcdtype = self.record_type(rcdid)
        kwds = self._check_fields(rcdtype, kwds)
        return _Las1V2_Xyz_Saving_Iterator(arr, rcdtype, self.scale, self.offset, fields=kwds)

    def header_type(self):
        return cfg2dtype(LasVersion1V2.Header)

    def _check_fields(self, rcdtype, kwds):
        _kwds = {}
        for k, v in kwds.items():
            if v is None:   continue
            assert k in rcdtype.names, 'invalid field (%s) for this record_type!' % k
            _kwds[k] = v
        return _kwds

    # -----------------------------
    def _validate_header(self):
        # assert '|'.join( '%s'%e for e in self._hdr['version'])=='1|2'
        assert self._hdr['dataformatid'] in self._RecordTypes.keys()

    def _read_more(self):
        N = self._hdr['offsetdata'] - self._hdr['headersize']
        self._vlrs = None if N == 0 else numpy.fromfile(self._src, dtype='b', count=N)

    def make_vlrs(self, name):
        return LasVersion1V2.VLR[name]

    def make_header(self, rcdid):
        assert rcdid >= 0, 'Record ID cannot be negative'
        assert rcdid in self._RecordTypes.keys(), 'Record ID does not exist in RecordTypes'

        h = numpy.zeros((1,), dtype=self.header_type())[0]
        h['filesignature'] = 'LASF'
        h['version'][0] = 1  # -- ! check it
        h['version'][1] = 2
        h['headersize'] = h['offsetdata'] = 227  # no vlrs
        h['scale'].fill(0.00025)  # -- ! check it

        h['dataformatid'] = rcdid  # -- record-format
        h['sizeofrecord'] = self.recordSizes[rcdid]
        h['softwareid'] = 'GreenValley'
        h['file_year'] = 2017
        h['systemid'] = 'EXTRACTION'
        return h

