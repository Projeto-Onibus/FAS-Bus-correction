#
# DataCorrection.py
# Descricao:
#

import configparser
import logging

import pandas as pd
import numpy as np
import cupy as cp

from LineDetection import HaversineLocal

def sigmoid(array):
	return 1 / (1 + np.exp(-array))

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
		CONFIGS['line_correction']['limit'] - The minimun value to consider a set of points as a valid group
	"""

	# Listagem com todos os conjuntos linha-ônibus detectados. Série contendo tuplas (<linha>,<sentido>,<onibus>)
	detections = detectionTable[detectionTable == True].stack()
	buses_detected = detections.index.get_level_values(2).unique()
	lines_detected = list(set(map(lambda x: (x[0], x[1]), detections.index)))
	busList = list(map(lambda x: x[0], busList))
	lineList = list(map(lambda x: (x[0]), lineList))

	correctedData = []
	for bus in buses_detected:
		busMap = np.array(busMatrix[busList.index(bus)])
		busMap = busMap[~np.any(np.isnan(busMap), axis=1)]					# Removendo valores NaN no fim do tensor
		lines = [i[0] for i in detectionTable[bus].loc[detectionTable[bus] == True].index.unique()]
		linesToCompare = []

		for line in lines:
			lineMap = np.array(linesMatrix[lineList.index(line)])
			lineMap = lineMap[~np.any(np.isnan(lineMap), axis=1)]
			distanceMatrix = sigmoid(int(CONFIGS["default_correction_method"]["distanceTolerance"]) 
										- cp.asnumpy(HaversineLocal(cp.asarray(np.expand_dims(busMap,0)), cp.asarray(np.expand_dims(lineMap,0)))[0]))
			distanceMatrix = np.squeeze(distanceMatrix)
			belongingArray = np.round(np.amax(distanceMatrix, axis=1))
			belongingArray = CorrectLine(belongingArray, CONFIGS['default_correction_method']['limit'])
			linesToCompare += [belongingArray]
		belongingMatrix = np.stack(linesToCompare)
		# Os pontos em que devemos nos preocupar são aqueles em que há sobreposição nas linhas, e reconhecendo estes pontos
		# podemos determinar o tamanho do grupo que estes pontos pertencem e usar este critério como abordagem para resolver o conflito
		conflicts1 = np.sum(belongingMatrix,axis=0)
		conflicts2 = np.nonzero(conflicts1 > 1)
		conflicts  = np.stack(conflicts2)
		#conflicts = np.stack(np.nonzero(np.sum(belongingMatrix,axis=0) > 1))

		if len(conflicts) != 0:
			# Matriz_prioridade consiste em uma matriz de M linhas onde M é o número de linhas de ônibus e N colunas onde N
			# é o número de pontos onde ocorreu conflito
			# A matriz_prioridade é preenchida com o número de pontos no grupo em que o conflito em um determinado ponto para
			# uma determinada linha foi detectado
			priorityMatrix = np.zeros((belongingMatrix.shape[0], conflicts.shape[0]))
			for line in range(len(belongingMatrix)):
				#ocurrences, counters = torch.unique_consecutive(belongingMatrix[line], return_counts=True)
				ocurrences = belongingMatrix[line][np.insert(np.absolute(np.diff(belongingMatrix[line])) > 0.00001, 0, True)]
				counters = np.diff(np.concatenate(([True],np.absolute(np.diff(belongingMatrix[line])) > 0,[True])).nonzero()[0])
				print("occurences")
				print(ocurrences)
				print("counters")
				print(counters)
				auxiliarCounters = np.cumsum(counters, 0)
				print("auxiliary counters")
				print(auxiliarCounters)
				print("belonging")
				print(belongingMatrix)
				for conflict in range(conflicts.shape[0]):
					if len(conflicts[conflict]) == 0:
						priorityMatrix[line][conflict] = 0
						break
					print("conflictsconfigl")
					print(conflicts)
					contagemConflitos = np.where(auxiliarCounters > conflicts[conflict])[0]
					print(contagemConflitos)
					if len(contagemConflitos) == 0:
						priorityMatrix[line][conflict] = 0
						break
					grupo = contagemConflitos.item()
					priorityMatrix[line][conflict] = counters[grupo] if ocurrences[grupo] == 1 else 0
			
			# As linhas cujo grupo que engloba o ponto de conflito é maior é selecionada para ser feita a substituição
			#_, dominantLines = priorityMatrix.max(0)
			dominantLines = priorityMatrix.max(0)

			dominantLines = dominantLines if (dominantLines.shape != tuple() and dominantLines.shape != (1,)) else dominantLines
			# Por fim a matriz de pertencimento é atualizada eliminando os conflitos
			for line in range(len(belongingMatrix)):
				for conflict in range(conflicts.shape[0]):
					if line != dominantLines[conflict]:
						belongingMatrix[line][conflicts[conflict]] = 0
			# Para evitar flutuações que possam surgir nesse processo os arrays de pertencimento de linha
			# passam novamente pela função CorrectLine()
			for line in range(len(belongingMatrix)):
				belongingMatrix[line] = CorrectLine(belongingMatrix[line], CONFIGS['default_correction_method']['limit'])
		correctedData += [[lines[i] for i in np.where(belongingMatrix == belongingMatrix.max(0))[0]]]
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
	#ocorrencias, contadores = torch.unique_consecutive(LineDetected, return_counts=True)
	ocorrencias = cp.asnumpy(LineDetected)[np.insert(np.absolute(np.diff(cp.asnumpy(LineDetected))) > 0.00001, 0, True)]
	contadores = np.diff(np.concatenate(([True],np.absolute(np.diff(cp.asnumpy(LineDetected))) > 0,[True])).nonzero()[0])

	# Array de contador auxiliar para saber onde alterar no array original
	contadores_auxiliar = contadores.cumsum()
	
	# Criar máscara para substituição no array de LineDetected
	mascara = np.zeros((LineDetected.shape[0]), dtype="bool")

	substituidos = np.where(contadores < int(limite))[0]
	if len(substituidos) != 0:
		for grupo in substituidos:
			if grupo == 0:
				inicio = contadores_auxiliar[0]
			else:
				inicio = contadores_auxiliar[grupo - 1]
			fim = inicio + contadores[grupo]
			mascara[inicio:fim] = True

	# Eliminar flutuações removendo grupos pequenos
	linha_Corrigida = np.where(mascara, np.logical_not(cp.asnumpy(LineDetected)).astype("int64"), cp.asnumpy(LineDetected).astype("int64"))
	
	return linha_Corrigida


if __name__ == "__main__":
	from time import time

	CONFIGS = configparser.ConfigParser()
	CONFIGS['default_correction_method'] = {'limit': '3', 'distanceTolerance': '300'}
	

	# Teste sanidade
#	 """
	matrizOnibus = pd.DataFrame(
		data={'M': np.array([1,1,0]), 'N': np.array([1,0,1]), 'O': np.array([0,1,0])},
		index=pd.MultiIndex.from_tuples([('B', '0'), ('A', '0'), ('A', '1')])
		)
	
	oni = np.array([[[1,0], [2,1], [3,3], [4,4], [2,4], [0,4], [float('NaN'),float('NaN')], [float('NaN'),float('NaN')]], [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8]], [[4,4], [4,3], [4,2], [4,1], [3,1], [2,1], [float('NaN'),float('NaN')], [float('NaN'),float('NaN')]]], dtype=float)
	li = np.array([[[0,0], [1,1], [2,2], [3,3], [4,4]], [[2,4], [1,4], [1,2], [1,1], [0,0]], [[4,5], [5,6], [6,7], [7,8], [8,8]]], dtype=float)
	busList = [(i, 0) for i in "M,N,O".split(",")]
	lineList = [('A','0'), ('B', '0'), ('A', '1')]

	print("antes:\n", matrizOnibus)
	print("oni:\n",oni)
	print("li:\n",li)
	resultado = CorrectData(matrizOnibus, oni, li, busList, lineList, CONFIGS)
	print("depois:\n", resultado.to_string())
#	 """

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
