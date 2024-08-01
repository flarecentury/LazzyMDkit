import itertools
import logging
import multiprocessing as mp
import pickle
from collections.abc import Container, Mapping
from concurrent import futures
from multiprocessing import Pool, cpu_count
from multiprocessing import Semaphore
from sys import getsizeof

import lz4.frame
import numpy as np
import pybase64
import tqdm

def pygrep(file, grep):
    with open(file,'r') as a:
        ls=a.readlines()
        filtered=[]
        for l in ls:
            if grep in l:
                filtered.append(l)
        print(''.join(filtered))


def pycat(file):
    with open(file,'r') as a:
        ls=a.read()
        print(ls)


def parallel_runs(func, data_lists, nproc=None):
    with futures.ProcessPoolExecutor(max_workers=nproc) as executor:
        result_list = executor.map(func, *data_lists)
    return result_list


def parallel_runs_mp(func, data_lists, parallel_mode=1, nproc=None):
    if nproc is None:
        nproc = cpu_count()
    mp_modes = ['fork', 'spawn', 'forkserver']
    mp_mode = mp_modes[parallel_mode]
    try:
        mp.set_start_method(mp_mode, force=True)
        print(mp_mode + 'ed')
    except RuntimeError:
        print('set mp mode failed')
        pass
    pool = mp.Pool(nproc)
    result_list = pool.starmap(func, data_lists)
    return result_list


class WriteBuffer:
    """Store a buffer for writing files.

    It is expensive to write to a file, so we need to make a buffer.
    
    Parameters
    ----------
    f: fileObject
        The file object to write.
    linenumber: int, default: 1200
        The number of contents to store in the buffer. The buffer will be flushed
        if it exceeds the set number.
    sep: str or bytes, default: None
        The separator for contents. If None (default), there will be no separator.
    """

    def __init__(self, f, linenumber=1200, sep=None):
        self.f = f
        if sep is not None:
            self.sep = sep
        elif f.mode == 'w':
            self.sep = ''
        elif f.mode == 'wb':
            self.sep = b''
        else:
            raise RuntimeError("File mode should be w or wb!")
        self.linenumber = linenumber
        self.buff = []
        self.name = self.f.name

    def append(self, text):
        """Append a text.
        
        Parameters
        ----------
        text: str
            The text to be appended.
        """
        self.buff.append(text)
        self.check()

    def extend(self, text):
        """Extend texts.

        Paramenters
        -----------
        text: list of strs
            Texts to be extended.
        """
        self.buff.extend(text)
        self.check()

    def check(self):
        """Check if the number of stored contents exceeds. 即未写到文件中的item数目超过self.linenumber时，则写入，来flush缓冲
        If so, the buffer will be flushed.
        """
        if len(self.buff) > self.linenumber:
            self.flush()

    def flush(self):
        """Flush the buffer."""
        if self.buff:
            self.f.writelines([self.sep.join(self.buff), self.sep])
            self.buff[:] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()
        self.f.__exit__(exc_type, exc_value, traceback)


def appendIfNotNone(f, wbytes):
    if wbytes is not None:
        f.append(wbytes)


def produce(semaphore, plist, parameter):
    """Item producer with a semaphore.
    """
    for item in plist:
        semaphore.acquire()
        if parameter is not None:
            item = (item, parameter)
        yield item


def compress(x, isbytes=False):
    """Compress the line.
    This function reduces IO overhead to speed up the program. The functions will
    use lz4 to compress and base64 to encode, since lz4 has better performance
    that any others.
    """
    if isbytes:
        return pybase64.b64encode(lz4.frame.compress(x, compression_level=0)) + b'\n'
    return pybase64.b64encode(lz4.frame.compress(x.encode(), compression_level=-1)) + b'\n'


def decompress(x, isbytes=False):
    """Decompress the line.
    """
    if isbytes:
        return lz4.frame.decompress(pybase64.b64decode(x.strip(), validate=True))
    return lz4.frame.decompress(pybase64.b64decode(x.strip(), validate=True)).decode()


def listtobytes(x):
    """Convert an object to a compressed line.
    """
    return compress(pickle.dumps(x), isbytes=True)


def bytestolist(x):
    """Convert a compressed line to an object.
    """
    return pickle.loads(decompress(x, isbytes=True))


def listtostirng(l, sep):
    """Convert a list to string, that is easier to store.

    Parameters
    ----------
    l: list of strs or lists
        The list to convert, which can contain any number of dimensions.
    sep: list of strs
        The seperators for each dimension.

    Returns
    -------
    str
        The converted string.
    """
    if isinstance(l, str):
        return l
    if isinstance(l, (list, tuple, np.ndarray)):
        return sep[0].join(map(lambda x: listtostirng(x, sep[1:]), l))
    return str(l)


def multiopen(pool, func, l, semaphore=None, nlines=None, unordered=True, return_num=False, start=0, extra=None,
              interval=None, bar=True, desc=None, unit="it", total=None):
    """Returns an interated object for process a file with multiple processors.

    Parameters
    ----------
    pool: multiprocessing.Pool
        The pool for multiprocessing.
    func: function
        The function to process lines.
    l: File object
        The file object.
    semaphore: multiprocessing.Semaphore, optional, default: None
        The semaphore to acquire. If None (default), the object will be passed
        without control.
    nlines: int, optional, default: None
        The number of lines to pass to the function each time. If None (default),
        only one line will be passed to the function.
    unordered: bool, optional, default: True
        Whether the process can be unordered.
    return_num: bool, optional, default: False
        If True, adds a counter to an iterable.
    start: int, optional, default: 0
        The start number of the counter.
    extra: object, optional, default: None
        The extra object passed to the item.
    interval: obj, optional, default: None
        The interval of items that will be passed to the function. For example,
        if set to 10, a item will be passed once every 10 items and others will
        be dropped.
    bar: bool, optional, default: True
        If True, show a tqdm bar for the iteration.
    desc: str, optional, default: None
        The description of the iteration shown in the bar.
    unit: str, optional, default: it
        The unit of the iteration shown in the bar.
    total: int, optional, default: None
        The total number of the iteration shown in the bar.
    
    Returns
    -------
    object
        An object that can be iterated.
    """
    obj = l
    if nlines:
        obj = itertools.zip_longest(*[obj] * nlines)
    if interval:
        obj = itertools.islice(obj, 0, None, interval)
    if return_num:
        obj = enumerate(obj, start)
    if semaphore:
        obj = produce(semaphore, obj, extra)
    if unordered:
        obj = pool.imap_unordered(func, obj, 100)
    else:
        obj = pool.imap(func, obj, 100)
    if bar:
        obj = tqdm.tqdm(obj, desc=desc, unit=unit, total=total)
    return obj


def run_mp(nproc, **arg):
    """Process a file with multiple processors.
    Parameters
    ----------
    nproc: int
        The number of processors to be used.
    Other parameters can be found in the `multiopen` function.
    """
    pool = Pool(nproc, maxtasksperchild=1000)
    semaphore = Semaphore(nproc * 150)
    try:
        results = multiopen(pool=pool, semaphore=semaphore, **arg)
        for item in results:
            yield item
            semaphore.release()
    except:
        logging.exception("run_mp failed")
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def must_be_list(obj):
    """Convert a object to a list if the object is not a list.
    """
    if isinstance(obj, list):
        return obj
    return [obj]


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
 
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, str):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r
