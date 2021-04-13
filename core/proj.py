#!coding=utf-8

import pyproj as pj
import numpy


_WGS84 = {
    'geocent': pj.Proj(proj='geocent', ellps='WGS84'),
    'longlat': pj.Proj(proj='latlong', ellps='WGS84'),
}

def epsg2proj4(epsg_code):
    crs = pj.CRS.from_epsg(epsg_code)
    return crs.to_proj4()


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
