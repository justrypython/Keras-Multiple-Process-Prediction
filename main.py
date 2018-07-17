from multiprocessing import Process, Queue
import os
import time
import argparse
from vgg16_worker import Vgg16Worker, Vgg16WorkerPool
from vgg16_worker import set_global_xnet, predict


class Scheduler:
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(Vgg16Worker(gpuid, self._queue))


    def start(self, xfilelst):

        # put all of files into queue
        for xfile in xfilelst:
            self._queue.put(xfile)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print "all of workers have been done"
        
class SchedulerPool(object):
    def __init__(self, gpuids):
        self._queue = Queue()
        self._data = []
        self._gpuids = gpuids
        
        self.p = None
        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._queue.put(gpuid)
        self.p = Vgg16WorkerPool(len(self._gpuids), set_global_xnet, [self._queue])


    def start(self, xfilelst):

        # put all of files into queue
        for xfile in xfilelst:
            self._data.append(xfile)
        
        st = time.time()
        r1 = self.p.map(predict, self._data)
        #self.p.close()
        #self.p.join()
        time.sleep(20)
        r2 = self.p.map(predict, self._data)
        self.p.close()
        self.p.join()
        et = time.time()
        print "all of workers have been done"

                

def run(img_path, gpuids):
    #scan all files under img_path
    xlist = list()
    for xfile in os.listdir(img_path):
        if '.jpg' in xfile:
            xlist.append(os.path.join(img_path, xfile))
    
    #init scheduler
    x = SchedulerPool(gpuids)
    
    #start processing and wait for complete 
    x.start(xlist)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", help="path to your images to be proceed")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )

    args = parser.parse_args()

    gpuids = [int(x) for x in args.gpuids.strip().split(',')]

    print args.imgpath
    print gpuids

    run(args.imgpath, gpuids)
    