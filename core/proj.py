#!coding=utf-8

import pyproj as pj
import numpy


_WGS84 = {
    'geocent': pj.Proj(proj='geocent', ellps='WGS84'),
    'longlat': pj.Proj(proj='latlong', ellps='WGS84'),
    # 'geocent-geoid':    pj.Proj(proj='latlong',ellps='WGS84', geoidgrids=_EGM96),
    # 'latlong-geoid':    pj.Proj(proj='latlong',ellps='WGS84', geoidgrids=_EGM96),
    # 'latlong-geoid2':   pj.Proj(proj='latlong',ellps='WGS84', geoidgrids=_EGM08),
}

def wgs84_to(x, y, z, proj='geocent', tm='EPSG:4326'):
    proj, tm = proj.lower(), tm.lower()
    # from IPython import embed; embed()
    assert _WGS84.__contains__(proj)
    p = pj.Proj(init=tm)
    rs = pj.transform(_WGS84[proj], p, x, y, z)
    return numpy.array(rs).T


def wgs84_from(e, n, z, proj='geocent', tm='EPSG:4326'):
    proj, tm = proj.lower(), tm.lower()
    p = pj.Proj(init=tm)
    rs = pj.transform(p, _WGS84[proj], e, n, z)
    return numpy.array(rs).T

def trans_wgs84(x, y, z):  # pop matrix:  prcs_xyz*M => wgs84_xyz
    # xyz should be wgs84
    arr = wgs84_to(x, y, z, tm='longlat')[:2]
    arr = numpy.radians(arr)  # long,lat
    sa, sb = numpy.sin(arr)
    ca, cb = numpy.cos(arr)
    return numpy.array([
        [-sa, -ca * sb, ca * cb, x],
        [ca, -sa * sb, sa * cb, y],
        [0, cb, sb, z],
        [0, 0, 0, 1]
    ]).T


def inv_wgs84(M):  # raw major, matrix: ecef->prcs
    inv_popM_R = M[:3, :3].T
    inv_popM_t = -M[3, :3].dot(inv_popM_R)
    invM = numpy.zeros_like(M)
    invM[:3, :3] = inv_popM_R
    invM[3, :3] = inv_popM_t
    invM[3, 3] = 1
    return invM

# --------------------------------------------
def prcs_from(arr, popM, tm='wgs84'):
    assert numpy.allclose(popM[:3, 3], 0), 'popM should be raw majar'
    if tm.lower() == 'prcs':
        return arr
    if arr.ndim == 1:
        if tm.lower() != 'wgs84':
            arr = wgs84_from(*arr, tm=tm)
        return numpy.dot((arr - popM[3, 0:3]), popM[0:3, 0:3].T)

    if tm.lower() != 'wgs84':
        arr = wgs84_from(arr[:, 0], arr[:, 1], arr[:, 2], tm=tm)
    # sR,sC =  popM.strides
    offset = popM[3, 0:3]  # as_strided( popM[3,0:3], shape=arr.shape,strides=(0,sC))
    return numpy.dot((arr - offset), popM[0:3, 0:3].T)


def prcs_to(arr, popM, tm='wgs84'):
    assert numpy.allclose(popM[:3, 3], 0), 'popM should be raw majar'
    if tm.lower() == 'prcs':  return arr

    if arr.ndim == 1:
        arr = numpy.dot(arr, popM[0:3, 0:3]) + popM[3, 0:3]
        if tm.lower() == 'wgs84': return arr
        return wgs84_to(*arr, tm=tm)
    # sR,sC =  popM.strides
    # offset = as_strided( popM[3,0:3], shape=arr.shape,strides=(0,sC))
    offset = popM[3, 0:3]
    arr = numpy.dot(arr, popM[0:3, 0:3]) + offset
    if tm.lower() == 'wgs84': return arr
    return wgs84_to(arr[:, 0], arr[:, 1], arr[:, 2], tm=tm)

def wgs84_trans_matrix(x, y, z):  # pop matrix:  prcs_xyz*M => wgs84_xyz
    # xyz should be wgs84
    arr = wgs84_to(x, y, z)[:2]
    arr = numpy.radians(arr)  # long,lat
    sa, sb = numpy.sin(arr)
    ca, cb = numpy.cos(arr)
    return numpy.array([
        [-sa, -ca * sb, ca * cb, x],
        [ca, -sa * sb, sa * cb, y],
        [0, cb, sb, z],
        [0, 0, 0, 1]
    ]).T
