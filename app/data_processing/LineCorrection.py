#
# DataCorrection.py
# Descricao:
#

import configparser
import logging

import pandas as pd
import numpy as np
import torch

def CorrectData(detectionTable, busMatrix, linesMatrix, busList, lineList, CONFIGS):
    """
    Arguments:
        detectionTable - detectionTable[('<linha>', '<direção>']['<ônibus>'] -> True se linha pertence ao ônibus
        busMatrix - busMatrix[<onibus>][<coord>][<lat/lon>]
        linesMatrix - linesMatrix[<linha>][<coord>][<lat/lon>]
        busList - busList[<indice>] -> (<identificadorOnibus>,<tamanho>)
        lineList -lineList[<indice>] -> (<identificadorLinha>,<direcao>)
        CONFIGS - configparser object
    Returns:
        results - Pandas DataFrame of same bus data with appended column of corrected line column data
    Description:
        Gets all of bus data and corrects each bus' line column based on results from FilterData
    Required Configurations:
        CONFIGS['lineCorrection']['limit'] - The minimun value to consider a set of points as a valid group
    """

    # Listagem com todos os conjuntos linha-ônibus detectados. Série contendo tuplas (<linha>,<sentido>,<onibus>)
    detections = detectionTable[detectionTable == True].stack()
    buses_detected = detections.index.get_level_values(2).unique()
    lines_detected = list(set(map(lambda x: (x[0], x[1]), detections.index)))
    busList = list(map(lambda x: x[0], busList))
    lineList = list(map(lambda x: (x[0]), lineList))

    correctedData = []
    i = 0
    for bus in buses_detected:
        busMap = torch.tensor(busMatrix[busList.index(bus)])
        busMap = busMap[~torch.any(busMap.isnan(), dim=1)]                  # Removendo valores NaN no fim do tensor
        lines = [i[0] for i in detectionTable[bus].loc[detectionTable[bus] == True].index.unique()]
        linesToCompare = []
        j = 0
        for line in lines:
            lineMap = torch.tensor(linesMatrix[lineList.index(line)])
            lineMap = lineMap[~torch.any(lineMap.isnan(), dim=1)]
            distanceMatrix = torch.sigmoid(int(CONFIGS["lineCorrection"]["distanceTolerance"]) - haversine(busMap, lineMap))
            belongingArray = torch.round(torch.max(distanceMatrix, dim=1)[0])
            belongingArray = CorrectLine(belongingArray, CONFIGS['lineCorrection']['limit'])
            linesToCompare += [belongingArray]
        belongingMatrix = torch.stack(linesToCompare)

        # Os pontos em que devemos nos preocupar são aqueles em que há sobreposição nas linhas, e reconhecendo estes pontos
        # podemos determinar o tamanho do grupo que estes pontos pertencem e usar este critério como abordagem para resolver o conflito
        conflicts = torch.nonzero(torch.sum(belongingMatrix, axis=0) > 1)
        #conflicts = conflicts.unsqueeze(dim=0) if conflicts.shape == torch.Size([]) else conflicts

        if len(conflicts) != 0:
            # Matriz_prioridade consiste em uma matriz de M linhas onde M é o número de linhas de ônibus e N colunas onde N
            # é o número de pontos onde ocorreu conflito
            # A matriz_prioridade é preenchida com o número de pontos no grupo em que o conflito em um determinado ponto para
            # uma determinada linha foi detectado
            priorityMatrix = torch.zeros(belongingMatrix.shape[0], conflicts.shape[0])
            for line in range(len(belongingMatrix)):
                ocurrences, counters = torch.unique_consecutive(belongingMatrix[line], return_counts=True)
                auxiliarCounters = torch.cumsum(counters, 0)
                for conflict in range(conflicts.shape[0]):
                    grupo = torch.where(auxiliarCounters > conflicts[conflict])[0][0].item()
                    priorityMatrix[line][conflict] = counters[grupo] if ocurrences[grupo] == 1 else 0
            
            # As linhas cujo grupo que engloba o ponto de conflito é maior é selecionada para ser feita a substituição
            _, dominantLines = torch.max(priorityMatrix, dim=0)
            dominantLines = dominantLines.squeeze(0) if (dominantLines.shape != torch.Size([]) and dominantLines.shape != torch.Size([1])) else dominantLines
            # Por fim a matriz de pertencimento é atualizada eliminando os conflitos
            for line in range(len(belongingMatrix)):
                for conflict in range(conflicts.shape[0]):
                    if line != dominantLines[conflict]:
                        belongingMatrix[line][conflicts[conflict]] = 0
            
            # Para evitar flutuações que possam surgir nesse processo os arrays de pertencimento de linha
            # passam novamente pela função CorrectLine()
            for line in range(len(belongingMatrix)):
                belongingMatrix[line] = CorrectLine(belongingMatrix[line], CONFIGS['lineCorrection']['limit'])
            correctedData += [[lines[i] for i in torch.max(belongingMatrix, dim=0)[1]]]
        else:
            correctedData += [[lines[i] for i in torch.max(belongingMatrix, dim=0)[1]]]
    # Criação de dataframe pandas. matriz de m linhas representando os ônibus e n colunas representando os pontos de ônibus.
    correctedDataframe = pd.DataFrame(correctedData, index=buses_detected)

    return correctedDataframe.T


def CorrectLine(LineDetected, limite):
    """
    The function receives a 'belonging array'  for one line and the minimum group limit and eliminate fluctuations.
    Arguments:
        LineDetected: 'belonging array' (torch tensor).
        limit: The minimun value to consider a set of points as a valid group
    Return: new array without fluctuations
    """
    
    # Resumimos a quantidade de informação utilizando o método unique_consecutive, criando o tensor ocorrencias com
    # as sequências de grupos e o tensor contadores com o número de ocorrências para o i-ésimo grupo
    ocorrencias, contadores = torch.unique_consecutive(LineDetected, return_counts=True)

    # Array de contador auxiliar para saber onde alterar no array original
    contadores_auxiliar = torch.cumsum(contadores, 0)
    # Criar máscara para substituição no array de LineDetected
    mascara = torch.zeros(contadores_auxiliar[-1], dtype=torch.bool)
    substituidos = torch.where(contadores < int(limite))[0]
    if len(substituidos) != 0:
        for grupo in substituidos:
            if grupo == 0:
                inicio = contadores_auxiliar[0]
            else:
                inicio = contadores_auxiliar[grupo - 1]
            fim = inicio + contadores[grupo]
            mascara[inicio:fim] = True
    
    # Eliminar flutuações removendo grupos pequenos
    linha_Corrigida = torch.where(mascara, torch.logical_not(LineDetected).long(), LineDetected.long())
    
    return linha_Corrigida

def haversine(busMap, lineMap):
    # Detection Algorithm

    # Separate latitudes from longitudes
    BusMapLat = busMap.select(1,0)
    BusMapLon = busMap.select(1,1)
    lineLat = lineMap.select(1,0)
    lineLon = lineMap.select(1,1)

    # Convert from decimal degrees to radians
    BusMapLat = torch.deg2rad(BusMapLat)
    BusMapLon = torch.deg2rad(BusMapLon)
    lineLat = torch.deg2rad(lineLat)
    lineLon = torch.deg2rad(lineLon)

    # Transpose Bus coordinates to orthogonal for broadcasting
    BusMapLat = torch.unsqueeze(BusMapLat, 1)
    BusMapLon = torch.unsqueeze(BusMapLon, 1)
    lineLat = torch.unsqueeze(lineLat, 0)
    lineLon = torch.unsqueeze(lineLon, 0)

    diffLat = BusMapLat - lineLat
    diffLon = BusMapLon - lineLon

    # Haversine
    d = (torch.sin(diffLat*0.5)**2 + torch.cos(BusMapLat) * torch.cos(lineLat) * torch.sin(diffLon*0.5)**2)
    results = 2 * 1000 * 6371.0088 * torch.asin(torch.sqrt(d))
    
    return results

if __name__ == "__main__":
    from time import time

    CONFIGS = configparser.ConfigParser()
    CONFIGS['lineCorrection'] = {'limit': '3', 'distanceTolerance': '300'}
    CONFIGS['lineDetection'] = {'distanceTolerance': 300}

    # Teste sanidade
#    """
    matrizOnibus = pd.DataFrame(
        data={'M': np.array([1,1,0]), 'N': np.array([1,0,1]), 'O': np.array([0,1,0])},
        index=pd.MultiIndex.from_tuples([('B', '0'), ('A', '0'), ('A', '1')])
        )
    
    oni = np.array([[[1,0], [2,1], [3,3], [4,4], [2,4], [0,4], [float('NaN'),float('NaN')], [float('NaN'),float('NaN')]], [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8]], [[4,4], [4,3], [4,2], [4,1], [3,1], [2,1], [float('NaN'),float('NaN')], [float('NaN'),float('NaN')]]], dtype=float)
    li = np.array([[[0,0], [1,1], [2,2], [3,3], [4,4]], [[2,4], [1,4], [1,2], [1,1], [0,0]], [[4,5], [5,6], [6,7], [7,8], [8,8]]], dtype=float)
    busList = [(i, 0) for i in "M,N,O".split(",")]
    lineList = [('A','0'), ('B', '0'), ('A', '1')]

    resultado = CorrectData(matrizOnibus, oni, li, busList, lineList, CONFIGS)
    print("antes:\n", matrizOnibus)
    print("depois:\n", resultado.to_string())
#    """

    # Teste desempenho
    """
    QO = 5000
    PPO = 3000
    QL = 2
    PPL = 3000
    oni = np.random.rand(QO,PPO,2) * 100
    li = np.random.rand(QL,PPL,2) * 100
    busList = ['O'+str(i) for i in range(QO)]
    lineList = [('L'+str(i),0) for i in range(QL)]
    
    matrizOnibus = pd.DataFrame((np.random.rand(QL,QO) > 0.5), index=pd.MultiIndex.from_tuples(lineList), columns=busList)

    print("Starting")
    start = time()
    resultado = CorrectData(matrizOnibus, oni, li, busList, lineList, CONFIGS)
    end = time()

    print(f"time:{end-start}")
    """