import cv2 as cv
import numpy as np
import math
import sys


class Candidate:

    def __init__(self, error, x_displacement, y_displacement):
        self.error = error
        self.y_displacement = y_displacement
        self.x_displacement = x_displacement



def generateCandidatesGrid(numCandidates,initialReferenceX , initialReferenceY):
    ORIGIN = 1
    grid =  np.zeros((numCandidates + ORIGIN, numCandidates + ORIGIN,2), int)
    coordX = 0
    coordY = 0
    if initialReferenceX > 0:
        for i in range(0, numCandidates + ORIGIN):
            if i < initialReferenceX:
                coordX =  (initialReferenceX- i)
            elif initialReferenceX - i <= 0:
                coordX =  -(i - initialReferenceX)
            if initialReferenceY > 0: 
                for j in range(0,numCandidates + ORIGIN):
                    if j < initialReferenceY:
                        coordY =  (initialReferenceY - j)
                    elif  initialReferenceY - j <= 0:
                        coordY =  -(j - initialReferenceY) 
                    grid.itemset((i,j,0), coordY) 
                    grid.itemset((i,j,1), coordX) 
            else:
                for j in range(0,numCandidates + ORIGIN):
                    if initialReferenceY + j <= 0:
                        coordY = (initialReferenceY + j)
                    elif  initialReferenceY + j > 0:
                        coordY =  (j + initialReferenceY) 
                    grid.itemset((i,j,0), coordY) 
                    grid.itemset((i,j,1), coordX) 
    else:
        for i in range(0, numCandidates + ORIGIN):
            if i > initialReferenceX:
                coordX =  (initialReferenceX + i)
            elif initialReferenceX - i <= 0:
                coordX =  -(i + initialReferenceX) 
            if initialReferenceY > 0: 
               for j in range(0,numCandidates + ORIGIN):
                    if j < initialReferenceY:
                        coordY =  (initialReferenceY - j)
                    elif  initialReferenceY - j <= 0:
                        coordY =  -(j - initialReferenceY) 
                    grid.itemset((i,j,0), coordY) 
                    grid.itemset((i,j,1), coordX) 
            else:
                for j in range(0,numCandidates + ORIGIN):
                    if initialReferenceY + j <= 0:
                        coordY = (initialReferenceY + j)
                    elif  initialReferenceY + j > 0:
                        coordY =  (j + initialReferenceY) 
                    grid.itemset((i,j,0), coordY) 
                    grid.itemset((i,j,1), coordX) 
    return grid



def bubbleSort(list):
    for passnum in range(len(list)-1,0,-1):
        for i in range(passnum):
            if list[i].error > list[i+1].error:
                temp = list[i]
                list[i] = list[i+1]
                list[i+1] = temp
    

def getMinErrorCandidate(candidates_error):
    optimal_motion = 0
    print(len(candidates_error))
    bubbleSort(candidates_error)
    optimal_candidate = candidates_error[0]
    return optimal_candidate


def computeCostFunction(referenceFrame, new_frame, phi):
    img_error = list()
    for i in range(0,referenceFrame.shape[0] - 1):
        for j in range(0,referenceFrame.shape[1] - 1):
            error =  phi(float(new_frame[i,j]) - float(referenceFrame[i, j]))      
            img_error.append(error)
    global_error = sum(img_error)
    return global_error


def printArray(array):
    for i in array:
        print(i)


def computeErrorCandidatesInGrid(candidatesGrid, phi, referenceFrame, new_frame):
    candidates_error = list()
    for i in range(0,candidatesGrid.shape[0]):
        for j in range(0,candidatesGrid.shape[1]):
                candidateX = candidatesGrid[i,j,0]
                candidateY = candidatesGrid[i,j,1]
                if (candidateX + 120 <= referenceFrame.shape[0] and candidateX + 120 >= 0) and (candidateY + 100 <= referenceFrame.shape[1] and candidateY + 100 >= 0):
                    referenceFrame = generateNextFrame(50, 100,70,120, candidateY, candidateX)                
                    error = computeCostFunction(referenceFrame, new_frame, phi)
                    candidate = Candidate(error, candidatesGrid[i,j,0], candidatesGrid[i,j,1])
                    print('Candidato: ',candidatesGrid[i,j,0], ' ' , candidatesGrid[i,j,1] )
                    print('Error: ', error)
                    candidates_error.append(candidate)
    return candidates_error


def printMatrix(matrix):
    print('Grid de candidatos: \n\n')
    for i in range(matrix.shape[0]):
        sys.stdout.write('\n')
        sys.stdout.flush()
        for j in range(matrix.shape[1]):
            sys.stdout.write('(')
            sys.stdout.flush()
            sys.stdout.write(str(matrix[i,j,0]))
            sys.stdout.flush()
            sys.stdout.write(',')
            sys.stdout.flush()
            sys.stdout.write(str(matrix[i,j,1]))
            sys.stdout.flush()         
            sys.stdout.write(')')
            sys.stdout.flush()
    print('\n')



def generateNextFrame(from_y, to_y, from_x, to_x, motionX, motionY):
    newFrame = cv.imread('black.jpeg',0)
    homogenize(newFrame,0)
    paintRectangle(newFrame, from_y + motionY, to_y + motionY, from_x + motionX, to_x + motionX, 255)
    return newFrame



def paintRectangle(img, from_y, to_y, from_x, to_x, color):
    for i in range(from_x, to_x):
        for j in range(from_y, to_y):
            img[i,j] = color


def homogenize(img, color):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = color


def computeError(candidates_grid, referenceFrame, new_frame):
    lorentz_w = .5
    print('Que función desea utilizar \n\n1. Cuadradado \n2. abs \n3. lorentziana con w = ' ,lorentz_w ,'\n\nIngresar una función usando los indices sin punto.')
    functionType = input('\n')
    candidates_error = list()
    if functionType == '1':
        square_func = lambda x: math.pow(x,2)
        candidates_error = computeErrorCandidatesInGrid(candidates_grid , square_func, referenceFrame, new_frame)
    elif functionType == '2':
         abs_func =  lambda x: abs(x)
         candidates_error = computeErrorCandidatesInGrid(candidates_grid , abs_func, referenceFrame, new_frame)
    elif functionType == '3':
        lorentzian_func = lambda x: math.log(1 + (pow(x,2) / 2 * pow(lorentz_w,2)))
        candidates_error = computeErrorCandidatesInGrid(candidates_grid , lorentzian_func, referenceFrame, new_frame)
    else:
        print('indice de función invalido')
        exit(0)

    return candidates_error



def main():
    
    referenceFrame = cv.imread('black.jpeg',0)
    homogenize(referenceFrame, 0)
    paintRectangle(referenceFrame,50, 100,70,120, 255)

    desplazamiento_x = int(input('introduzca un desplazamiento en X (solo enteros)\n'))
    desplazamiento_y = int(input('introduzca un desplazamiento en Y (solo enteros)\n'))

    new_frame = None
    if (desplazamiento_x + 120 <= referenceFrame.shape[0] and desplazamiento_x + 120 >= 0) and (desplazamiento_y + 100 <= referenceFrame.shape[1] and desplazamiento_y + 100 >= 0):
            new_frame = generateNextFrame(50, 100,70,120, desplazamiento_y, desplazamiento_x)
    else:
        print('displacement is out of boundaries!')
        exit(0)

        
    cv.imwrite('ref.png', referenceFrame)
    cv.imwrite('moving.png', new_frame)
    

    numCandidates = int(input('introduzca un numero de candidatos (solo pares)\n'))
    if numCandidates%2 != 0: print('Cantidad invalida de candidatos') , exit(0)
    referencePointY = int(input('introduzca un punto de referencia para el grid en el eje X\n'))
    referencePointX = int(input('introduzca un punto de referencia para el grid en el eje Y\n'))
    if (abs(referencePointX*2) + 1) < numCandidates + 1 or (abs(referencePointY) *2 + 1) < numCandidates + 1: print("\nERROR: demasiados candidatos para el punto de referencia provisto"), exit(0)

    candidates_grid = generateCandidatesGrid(numCandidates, referencePointX, referencePointY)
    printMatrix(candidates_grid)

    candidates_error = computeError(candidates_grid, referenceFrame, new_frame)
    optimal_candidate= getMinErrorCandidate(candidates_error)

    print('\n\nEl valor de desplazamiento mas optimo es: ', optimal_candidate.error , 'dado por el candidato: ', '(', optimal_candidate.x_displacement,',', optimal_candidate.y_displacement, ')')

    

main()