"""
this module provide functions for reading and writing data text files.
"""
import sys
from numpy import array

def makedir(foldername):
    import os
    import os.path
    if not os.path.isdir(foldername):
        os.makedirs(foldername)

def err(s):
    print(s, file=sys.stderr)

def errS(s):
    print(s, file=sys.stderr)
    sys.exit(1)

def openwritefile(filename):
    try:
        f = open(filename, 'w')
    except:
        print('cannot open file {0} for writing'.format(filename), file=sys.stderr)
        sys.exit(1)
    return f

def openreadfile(filename):
    try:
        f = open(filename, 'r')
    except:
        print('cannot open file {0} for reading'.format(filename), file=sys.stderr)
        sys.exit(1)
    return f

def readLines(filename, func):
    """
    filename contains some information, that can be parsed via func.
    """
    res = []
    f = openreadfile(filename)
    try:
        for line in f:
            res.append(func(line))
    except:
        print('cannot read from {0}:'.format(filename), file=sys.stderr)
        print(line, file=sys.stderr)
        exit(1)
    f.close()
    return array(res)

def readScalars(filename):
    def readScalar(s):
        return float(s)
    return readLines(filename, readScalar)

def readPoints(filename):
    def readPoint(s):
        return [float(h) for h in s.split()]
    return readLines(filename, readPoint)

def readField(filename):
    res = readPoints(filename)
    pos = res[:, 0:3]
    vec = res[:, 3:6]
    return (pos, vec)

def readMatrices(filename):
    def readMatrix(s):
        h = [float(hh) for hh in s.plit()]
        return array([[h[0], h[3], h[6]],
                      [h[1], h[4], h[7]],
                      [h[2], h[5], h[8]]])
    return readLines(filename, readMatrix)

def writeLines(filename, datas, func):
    f = openwritefile(filename)
    showLines(datas, func, f)
    f.close()

def writePoints(filename, points):
    def showPoint(p):
        return '{0[0]} {0[1]} {0[2]}'.format(p)
    writeLines(filename, points, showPoint)

def showFieldComponent(d):
    p = d[0]
    v = d[1]
    return '{0[0]} {0[1]} {0[2]} {1[0]} {1[1]} {1[2]}'.format(p, v)

def writeField(filename, pos, vec):
    arg = [[p[0], p[1], p[2], v[0], v[1], v[2]] for (p, v) in zip(pos, vec)]
    writeLines(filename, zip(pos, vec), showFieldComponent)


def showLines(datas, func, handle=sys.stdout):
    try:
        for d in datas:
            handle.write(func(d) + '\n')
    except:
        print('cannot write to {0}:'.format(filename), file=sys.stderr)
        print(repr(d), file=sys.stderr)
        exit(1)

def showField(pos, vec):
    arg = [[p[0], p[1], p[2], v[0], v[1], v[2]] for (p, v) in zip(pos, vec)]
    showLines(zip(pos, vec), showFieldComponent)
