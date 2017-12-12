"""
    Utility to load minibatches in parallel

    with simple exemple class to usage

    date : 2016-10-20
"""
__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import time
import numpy as np
from PIL import Image
import abc
from multiprocessing import Process, Queue, cpu_count, JoinableQueue


class ParallelMinibatch:
    def __init__(self, max_size=0):
        self.max_size = max_size
        self.N_Process = cpu_count()
        self.tasks = None
        self.results = None
        self.processes = []

    def __enter__(self):
        self.init_processes()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_processes()

    def init_processes(self):
        if self.processes:
            raise Exception("init_processes is called but there are still running process!")
        self.tasks = Queue()
        self.results = Queue(self.max_size)

        self.minibatches_indexes = self.compute_minibatches_permutations_()
        self.task_qty = len(self.minibatches_indexes)
        for task in self.minibatches_indexes:
            self.tasks.put(task)

        self.processes = [Process(target=self.worker_, args=(self.results, self.tasks)) for i in range(self.N_Process)]
        for proc in self.processes:
            proc.start()

    def stop_processes(self):
        if not self.processes:
            raise Exception("stop_processes is called but there are no process running!")
        for i in range(self.N_Process):
            self.tasks.put(None)
        # todo implement a way to remove current task so we do not have to wait for all task to be completed
        for proc in self.processes:
            proc.join()
        self.tasks = None
        self.results = None
        self.processes = []

    def worker_(self, results, tasks):
        while True:
            task = tasks.get(block=True, timeout=None)
            if task is None:
                break
            batch = self.load_minibatch(task)
            results.put(batch)

    def get_minibatch(self):
        """
        This function block until the next task is ready
        :return:
        """
        if not self.processes:
            raise Exception("init_processes before getting minibatches")
        for i in range(self.task_qty):
            result = self.results.get(block=True, timeout=None)
            yield result
        self.task_qty = 0

    @abc.abstractmethod
    def load_minibatch(self, task):
        """
        User implementation of minibatch load code. Task is a task given to input_tasks previously, return the minibatch
        the object will make sure to pass it safely to the consumer
        :param task:
        :return:
        """
        return

    @abc.abstractmethod
    def compute_minibatches_permutations_(self):
        """
        User implementation of minibatch permutations. It should return a list of list of index :
        [[6, 1, 3], [2, 0, 9], [8, 5, 7]]
        :return:
        """
        return


class ExempleMinibatchLoader(ParallelMinibatch):
    def load_minibatch(self, task):
        """
        Simple exemple implementation of loading/worker code
        :param task:
        :return:
        """
        path = "/home/mathieu/Dataset/DeepTrack/raw_bunny/test_random/"
        for index in task:
            image = np.array(Image.open(path + str(index) + ".png"))
        return image

    def compute_minibatches_permutations_(self):
        # For this exemple I generate a list of index permutations, and split them in minibatches of 100
        permutations = np.random.permutation(range(7000))
        chunks = [permutations[x:x + 100] for x in range(0, len(permutations), 100)]
        return chunks


if __name__ == '__main__':
    """
    Simple exemple comparing sequential code and parallel code. The result can be taken seriously only if you run this
    multiple times (timing may change while the cache get hot)
    todo : iterate the test multiple times to remove hot cache bias
    """

    # Instantiate the object, you can setup the max minibatch queue
    loader = ExempleMinibatchLoader()

    # This is the sequential code, which call load-minibatch on each "tasks"
    start_time = time.time()
    chunks = loader.compute_minibatches_permutations_()
    for chunk in chunks:
        loader.load_minibatch(chunk)
    print("Sequential : {}".format(time.time() - start_time))

    # This is the actual use of Parallel loading split on N processes.
    start_time = time.time()
    # here you can use with to start/end processes or use init_processes and stop_processes
    with loader:
        # retrieve the minibatch generator
        minibatches = loader.get_minibatch()
        # loop to get all results
        for i in minibatches:
            # process the minibatch
            pass

    print("Parallel : {}".format(time.time() - start_time))
