import time


# Class to hold performance and measure execution time 
# 
# TODO: Use destructor to save progress in case of raised exception
#

class Measure:
    def __init__(self, filenameInput):
        self.filename = filenameInput
        self.times = dict()
        self.times['all'] = { 'start':time.time(),'forced':False}

    def start(self,key):
        if not key in self.times.keys():
            self.times[key] = {'start':None,'end':None}
        self.times[key]['start'] = time.time()
        self.times[key]['forced'] = False

    def end(self,key):
        if not key in self.times.keys():
            raise Exception(f"Time for '{key}' not started")
        self.times[key]['end'] = time.time()

    def measure(self,key):
        self.times['all']['end'] = time.time()
        if not key in self.times.keys():
            raise Exception(f"No measure for '{key}'")
        if not self.times[key]['end']:
            self.end(key)
            self.times[key]['forced'] = True
        return self.times[key]['end'] - self.times[key]['start']
  
    def execute(self,func,args,title="function"):
        res = title + "\n"
        for key,_ in self.times.items():
            res += f"\t{key}: {func(self.measure(key),*args)}" + (" FORCED" if self.times[key]['forced'] else "") + "\n"
        return res

    def __str__(self):
        res = "Time results:\n"
        for key,_ in self.times.items():
            res += f"\t{key}: {self.measure(key)}" + (" FORCED" if self.times[key]['forced'] else "") + "\n"
        return res

    def __del__(self):
        for item in  self.times.keys():
            if not "end" in self.times[item]:
                self.end(item)
        with open(self.filename,"w") as fil:
            fil.write(f"{self}")
