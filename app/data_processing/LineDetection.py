import cupy as cp
import numpy as np
import pandas as pd

# TODO: ENVIAR TODA A MATRIZ PRA GPU E RETIRAR OS RESULTADOS POR LOOP

def FilterData(matrizOnibusCpu,matrizLinhasCpu,busIdList,lineIdList,CONFIGS,logging):
        
        distanceTolerance = float(CONFIGS['default_correction_method']['distanceTolerance'])
        detectionPercentage = float(CONFIGS['default_correction_method']['detectionPercentage'])
        
        busStepSize = int(CONFIGS['default_correction_method']['busStepSize'])
        lineStepSize = int(CONFIGS['default_correction_method']['lineStepSize'])
        
        matrizOnibus = cp.asarray(matrizOnibusCpu)
        matrizLinhas = cp.asarray(matrizLinhasCpu)

        busesList = cp.array_split(matrizOnibus,int(matrizOnibus.shape[0]/busStepSize if matrizOnibus.shape[0] > busStepSize else 1))
        for index in range(1,len(busesList)):
                if len(busesList[index].shape) == 2:
                        busesList[index][None,:]
                busesList[index] = cp.hsplit(busesList[index],[int(cp.argwhere(cp.isnan(busesList[index][0,:,1]))[0]),busesList[index].shape[1]])[0]
        

        linesList = cp.array_split(matrizLinhas,int(matrizLinhas.shape[0]/lineStepSize if matrizLinhas.shape[0] > lineStepSize else 1))
        for index in range(1,len(linesList)):
                if len(linesList[index].shape) == 2:
                        linesList[index][None,:]
                linesList[index] = cp.hsplit(linesList[index],[int(cp.argwhere(cp.isnan(linesList[index][0,:,1]))[0]),linesList[index].shape[1]])[0]


        fullResults = None
        for nowBus, busTensor in enumerate(busesList):
                allLinesResult = None
                for nowLine,lineTensor in enumerate(linesList):
                        algRes = cp.asnumpy(Algorithm(busTensor,lineTensor,TOLERANCE=distanceTolerance))
                        # Concatenation to full results matrix
                        if allLinesResult is None:
                                allLinesResult = np.copy(algRes)
                        else:
                                allLinesResult = np.concatenate([allLinesResult,algRes],axis=1)
                if fullResults is None:
                        fullResults = np.copy(allLinesResult)        
                else:
                        fullResults = np.concatenate([fullResults,allLinesResult],axis=0)
        lineLabel = [(i[0],str(i[1])) for i in lineIdList]
        busLabel = [i[0] for i in busIdList]
        results = pd.DataFrame(fullResults.T,index=pd.MultiIndex.from_tuples(lineLabel),columns=busLabel)
        return results


def Algorithm(MO,ML,TOLERANCE,detectionPercentage=None,haversine=True):
    results = HaversineLocal(MO,ML)
    
    # Matriz D^[min]
    resultsMin = cp.nanmin(results,axis=1)
    sizeLine = resultsMin.shape[2]
    below = cp.sum(resultsMin<TOLERANCE,axis=2)
    resultsPerc = below / (sizeLine - infVector)
    if detectionPercentage:
        return resultsPerc > cp.array(detectionPercentage)
    return resultsPerc

def HaversineLocal(busMatrix,lineMatrix):
    MatrizOnibus = cp.copy(MO)
    MatrizLinhas = cp.copy(ML)

    MatrizLinhas = cp.dsplit(MatrizLinhas,2)
    MatrizOnibus = cp.dsplit(MatrizOnibus,2)

    infVector = cp.squeeze(cp.sum(cp.isnan(MatrizLinhas[0]),axis=1),axis=-1)

    MatrizLinhas[0] = cp.expand_dims(MatrizLinhas[0],axis=-1)
    MatrizLinhas[1] = cp.expand_dims(MatrizLinhas[1],axis=-1)
    MatrizOnibus[0] = cp.expand_dims(MatrizOnibus[0],axis=-1)
    MatrizOnibus[1] = cp.expand_dims(MatrizOnibus[1],axis=-1)

    MatrizOnibus[0] *=  cp.pi/180 
    MatrizOnibus[1] *=  cp.pi/180
    MatrizLinhas[1]  =  cp.transpose(MatrizLinhas[1],[2,3,0,1]) * cp.pi/180
    MatrizLinhas[0]  =  cp.transpose(MatrizLinhas[0],[2,3,0,1]) * cp.pi/180

    # Haversine or euclidian, based on <haversine>
    if haversine:
        results = 1000*2*6371.0088*cp.arcsin(
        cp.sqrt(
            (cp.sin((MatrizOnibus[0] - MatrizLinhas[0])*0.5)**2 + \
             cp.cos(MatrizOnibus[0])* cp.cos(MatrizLinhas[0]) * cp.sin((MatrizOnibus[1] - MatrizLinhas[1])*0.5)**2)
        ))
    else:
        results = cp.sqrt((MatrizOnibus[0]-MatrizLinhas[0])**2+(MatrizOnibus[1]-MatrizLinhas[1])**2)

    return results


if __name__ == '__main__':
        from configparser import ConfigParser
        from time import time
        import logging

        CONFIG = ConfigParser()

        # Parametros de execucao
        CONFIG.add_section('default_correction_method')
        CONFIG['default_correction_method']['busStepSize'] = '5'
        CONFIG['default_correction_method']['lineStepSize'] = '5'
        CONFIG['default_correction_method']['distanceTolerance'] = '5000'
        CONFIG['default_correction_method']['detectionPercentage'] = '0.5'
        
        
        # Teste de desempenho
        quantOni = 40
        quantLi = 20
        oni = np.random.rand(quantOni,2000,2)
        oni = np.pad(oni,[(0,0),(0,4),(0,0)],constant_values=np.NaN)
        li = np.random.rand(quantLi,2000,2)
        li = np.pad(li,[(0,0),(0,4),(0,0)],constant_values=np.NaN)


        busList = [f"O-{i}" for i in range(quantOni)]
        lineList = [f"L-{i}" for i in range(quantLi)]


        if not CONFIG.has_section('default_correction_method'):
                CONFIG['default_correction_method'] = {}

        start = time()
        res = FilterData(oni,li,busList,lineList,CONFIG,logging)
        end = time()
        print(f"{quantOni}@2000 X {quantLi}@2000",end-start)
        #print(results)