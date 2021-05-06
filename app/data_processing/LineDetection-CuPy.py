import numpy as np 
import pandas as pd

import cupy as cp

def FilterData(matrizOnibus,matrizLinhas,busIdList,lineIdList,CONFIGS,logging):
        
        distanceTolerance = float(CONFIGS['lineDetection']['distanceTolerance'])
        detectionPercentage = float(CONFIGS['lineDetection']['detectionPercentage'])
        
        busStepSize = int(CONFIGS['lineDetection']['busStepSize'])
        lineStepSize = int(CONFIGS['lineDetection']['lineStepSize'])
        
        #busDataset = tr.utils.data.DataLoader(matrizOnibus,batch_size=busStepSize,drop_last=True)
        #lineDataset = tr.utils.data.DataLoader(matrizLinhas,batch_size=lineStepSize,drop_last=True)

        busDataset = np.array_split(matrizOnibus,busStepSize)
        lineDataset = np.array_split(matrizLinhas,lineStepSize)

        #algorithm = Algorithm()
        #algorithm.cuda()

        fullResults = None
        
        for busTensor in busDataset:
                allLinesResult = None
                #busTensor = busTensor.cuda()
                for lineTensor in lineDataset:
                        # Algorithm segment
                        #lineTensor = lineTensor.cuda()
                        segmentResults = cp.asnumpy(FullAlgorithm(cp.asarray(busTensor),cp.asarray(lineTensor),distanceTolerance,detectionPercentage))
                        # Concatenation to full results matrix
                        if allLinesResult is None:
                                allLinesResult = np.copy(segmentResults)
                        else:
                                allLinesResult = np.concatenate([allLinesResult,segmentResults],axis=1)
                if fullResults is None:
                        fullResults = np.copy(allLinesResult)        
                else:
                        fullResults = np.concatenate([fullResults,allLinesResult],axis=0)

        fullResults = fullResults > detectionPercentage
        lineLabel = [(i[0],str(i[1])) for i in lineIdList]
        busLabel = [i[0] for i in busIdList]
        results = pd.DataFrame(fullResults.T,index=pd.MultiIndex.from_tuples(lineLabel),columns=busLabel)
        return results

                
def FullAlgorithm(MatrizOnibus,MatrizLinhas,TOLERANCE,detectionPercentage):
        
        MatrizLinhas = list(cp.chunk(MatrizLinhas,2,dim=2))
        MatrizOnibus = list(cp.chunk(MatrizOnibus,2,dim=2))
        
        MatrizLinhas[0].unsqueeze_(3)
        MatrizLinhas[1].unsqueeze_(3)
        MatrizOnibus[0].unsqueeze_(3)
        MatrizOnibus[1].unsqueeze_(3)

        MatrizOnibus[0] *=  np.pi/180 
        MatrizOnibus[1] *=  np.pi/180
        MatrizLinhas[1] = cp.movedim(MatrizLinhas[1],(0,1),(2,3)) * np.pi/180
        MatrizLinhas[0] = cp.movedim(MatrizLinhas[0],(0,1),(2,3)) * np.pi/180

        # MAtriz D
        results = 1000*2*6371.0088*cp.asin(cp.sqrt(
        (cp.sin((MatrizOnibus[0] - MatrizLinhas[0])*0.5)**2 + cp.cos(MatrizOnibus[0])* cp.cos(MatrizLinhas[0]) * cp.sin((MatrizOnibus[1] - MatrizLinhas[1])*0.5)**2)
        ))

        # Matriz D^[min]
        results = cp.amin(results,dim=1)

        sizeLine = results.shape[2]
        infMatrix = cp.sum(cp.isinf(results) + cp.isnan(results),axis=2)
        below = cp.sum(results<TOLERANCE,axis=2)
        results = below / (sizeLine - infMatrix)
        return results




if __name__ == '__main__':
        from configparser import ConfigParser
        from time import time
        import logging

        CONFIG = ConfigParser()

        # Parametros de execucao
        CONFIG.add_section('lineDetection')
        CONFIG['lineDetection']['busSize'] = '2'
        CONFIG['lineDetection']['lineSize'] = '2'
        CONFIG['lineDetection']['distanceTolerance'] = '2000'
        CONFIG['lineDetection']['detectionPercentage'] = '0.5'
        
        # Teste de desempenho
        np.random.default_rng()
        oni = np.random.rand(10,5000,2)
        li = np.random.rand(10,5000,2)


        busList = [(i,) for i in "A,B,C,D,E,F,G,H,I,J".split(",")]
        lineList = [(i,) for i in "M,N,O,P,Q,R,S,T,U,V".split(",")]
        

        if not CONFIG.has_section('lineDetection'):
                CONFIG['lineDetection'] = {}
        
        start = time()
        results = FilterData(oni,li,busList,lineList,CONFIG,logging)
        end = time()
        print("perf:",end-start)
        print(results)