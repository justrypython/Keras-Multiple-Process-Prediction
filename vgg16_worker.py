from multiprocessing import Queue, Process
from multiprocessing.pool import Pool
import cv2
import numpy as np
import os

xnet = None

def set_global_xnet(gpuids):
    #from wingdebugger import wingdbstub
    #wingdbstub.Ensure()
    #wingdbstub.debugger.StartDebug()
    global xnet
    gpuid = gpuids.get()
    print('#############################', gpuids)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    import vgg16
    xnet = vgg16.Vgg16('/media/zhaoke/b0685ee4-63e3-4691-ae02-feceacff6996/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    #wingdbstub.debugger.StopDebug()
    
def predict(imgfile):
    #BGR
    im = cv2.resize(cv2.imread(imgfile), (224, 224)).astype(np.float32)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = im.reshape((1, 224, 224, 3))
    out = xnet.predict(im)
    print ' xfile ', imgfile#, " predicted as label", out
    return np.argmax(out)   
    
RUN = 0
class Vgg16WorkerPool(Pool):
    def __init__(self, processes, initializer=set_global_xnet, initargs=None):
        super(Vgg16WorkerPool, self).__init__(processes=processes,
                                              initializer=initializer,
                                              initargs=initargs)
        
    def map(self, func, iterable, chunksize=None):
        '''
        Equivalent of `map()` builtin
        '''
        #from wingdebugger import wingdbstub
        #wingdbstub.Ensure()
        #wingdbstub.debugger.StartDebug()
        assert self._state == RUN
        ret = self.map_async(func, iterable, chunksize).get()
        #wingdbstub.debugger.StopDebug()
        return ret

        

class Vgg16Worker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue

    def run(self):

        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        #load models
        import wingdbstub
        wingdbstub.Ensure()
        wingdbstub.debugger.StartDebug()
        import vgg16
        #download the vgg16 weights from
        #https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
        xnet = vgg16.Vgg16('/media/zhaoke/b0685ee4-63e3-4691-ae02-feceacff6996/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

        print 'vggnet init done', self._gpuid

        while True:
            xfile = self._queue.get()
            if xfile == None:
                self._queue.put(None)
                break
            label = self.predict(xnet, xfile)
            print 'woker', self._gpuid, ' xfile ', xfile, " predicted as label", label

        print 'vggnet done ', self._gpuid
        wingdbstub.debugger.StopDebug()

    def predict(self, xnet, imgfile):
        #BGR
        im = cv2.resize(cv2.imread(imgfile), (224, 224)).astype(np.float32)

        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68

        im = im.reshape((1, 224, 224, 3))
        out = xnet.predict(im)
        return np.argmax(out)

