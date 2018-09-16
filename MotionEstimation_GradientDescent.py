
import cv2 as cv
import numpy as np
import math
from scipy import interpolate
import sys


def computeGradientDescent(epochs, epsilon, alpha, reference_frame, new_frame, initialSolutionX, initialSolutionY):
    print('\n\nPreparing Gradient descent...')
    print("\nMax_Epochs = ", epochs)
    print("\nAlpha = ",alpha)
    print("\nEpsilon = ",epsilon)
    print('\n-------------------------------------------------------------')
    print('\n\n----------------Initializing Hell Descent...---------------')
    print('\n\n-----------------------------------------------------------')
    print('\n“Hope not ever to see Heaven. I have come to lead you to the other shore; into eternal darkness; into fire and into ice.” ― Dante Alighieri, Inferno ')
    input("\n\nPress Enter to continue...")
    print('\n\n\nAbandon all hope — Ye Who Enter Here\n\n')
    square_func = lambda x: math.pow(x,2)
    motionX = initialSolutionX
    motionY = initialSolutionY
    for i in range(0, epochs):
        reference_frame = applyDisplacement(reference_frame, motionX, motionY)
        costGradient = computeCostFunctionGradient(reference_frame, new_frame, square_func, motionX, motionY)
        costX = costGradient[0] * alpha
        costY = costGradient[1] * alpha
        old_vX = motionX
        old_vY = motionY
        motionX = old_vX - costX
        motionY = old_vY - costY
        residualX = np.linalg.norm(motionX - old_vX)
        residualY = np.linalg.norm(motionY - old_vY)
        print('\nStep: ',i + 1)
        print('\n\nCost = ',motionX,' ', motionY)
        if(residualX < epsilon and residualY < epsilon):
            print('\n\n\nFunction is now "epsilon exhausted"')
            print('Optimization is over.')
            break;
    print('\n\n--------------------------------------------------------------')
    print('\n\n--------------------------------------------------------------')
    print('\n\n-----------------------------------------------------------\n\n')
    print('“From there we came outside and saw the stars..."― Dante Alighieri, Inferno \n')
    print('Done.')
    motion_vector = (motionX, motionY)
    return motion_vector


def computeCostFunctionGradient(reference_frame, new_frame, phi, motion_gradientX, motion_gradientY):
    img_errorX = list()
    img_errorY = list()
    for i in range(2,reference_frame.shape[0] - 2):
        for j in range(2,reference_frame.shape[1] - 2):
            time_derivative = computeTimeDerivative(new_frame, reference_frame, i, j)
            dotProduct_d = computeSpatialDerivativeX(reference_frame, i,j) * motion_gradientX + computeSpatialDerivativeY(reference_frame, i,j) * motion_gradientY
            gradientX = ((time_derivative - dotProduct_d) * -computeSpatialDerivativeX(reference_frame, i,j))  
            gradientY = ((time_derivative - dotProduct_d) * -computeSpatialDerivativeY(reference_frame, i,j)) 
            img_errorX.append(gradientX)
            img_errorY.append(gradientY)
    gradientX = sum(img_errorX)
    gradientY = sum(img_errorY)
    gradient_vector = (gradientX, gradientY)
    return gradient_vector


def computeTimeDerivative(current_img, previous_img, coordX, coordY):
    return float(current_img[coordX,coordY]) - float(previous_img[coordX, coordY])

def computeSpatialDerivativeX(img, coordX, coordY):
    return img[coordX,coordY] - img[coordX - 1, coordY]


def computeSpatialDerivativeY(img, coordX, coordY):
    return img[coordX,coordY] - img[coordX, coordY - 1]


def computeTimeDerivativeMatrix(reference_frame, new_frame):
    time_derivative_matrix = np.zeros((reference_frame.shape[0], reference_frame.shape[1],1) , float)
    for i in range(0, reference_frame.shape[0]):
        for j in range(0, reference_frame.shape[1]):
            time_derivative_matrix[i,j] = computeTimeDerivative(new_frame, reference_frame, i, j)
    return time_derivative_matrix


def computeGradients(reference_frame, new_frame):
    gradient_matrix = np.zeros((reference_frame.shape[0],reference_frame.shape[1],2), float)
    for i in range(1,reference_frame.shape[0]):
        for j in range(1, reference_frame.shape[1]):
            spatialX = reference_frame[i,j] - reference_frame[i - 1, j]
            spatialY =  reference_frame[i,j] -   reference_frame[i, j - 1]
            gradient_matrix.itemset((i,j, 0), spatialX) 
            gradient_matrix.itemset((i,j, 1), spatialY)
    return gradient_matrix


def matrix_square_sum(matrix, channel):
    x_axis = np.zeros(matrix.shape[0], float)
    for i in range(matrix.shape[0]):
        y_sum = 0
        for j in range(matrix.shape[1]):
            y_sum = y_sum + math.pow(matrix[i,j,channel],2)   
        x_axis[i] = y_sum
    return float(sum(x_axis))


def interpolation2d(grid):
    x_coords = np.arange(0, grid.shape[0])
    y_coords = np.arange(0, grid.shape[1])
    return interpolate.interp2d(y_coords, x_coords, grid, kind = "cubic")
    


def matrix_product_sumation(matrix):
    x_axis = np.zeros(matrix.shape[0], float)
    for i in range(matrix.shape[0]):
        y_sum = 0
        for j in range(matrix.shape[1]):
            spatialX = matrix[i,j,0]
            spatialY = matrix[i,j,1]
            y_sum =  y_sum + ( spatialX *  spatialY) 
        x_axis[i] = y_sum
    product_sum = float(sum(x_axis))
    return product_sum


def twoMatrix_product_sumation(matrixA, matrixB, matrixA_channel):
    x_axis = np.zeros(matrixA.shape[0],float)
    for i in range(matrixA.shape[0]):
        y_sum = 0
        for j in range(matrixA.shape[1]):
            y_sum = y_sum + matrixA[i,j,matrixA_channel] *  matrixB[i,j] 
        x_axis[i] = y_sum
    return float(sum(x_axis))


def ensemble_matrix_A(gradient_matrix):
    A = np.zeros((2,2), float)
    A[0,0] = matrix_square_sum(gradient_matrix, 0)
    A[0,1] =  matrix_product_sumation(gradient_matrix)
    A[1,0] =  A[0,1]
    A [1,1] = matrix_square_sum(gradient_matrix, 1)
    return A


def compute_vector_b(gradient_matrix, time_derivative):
    b1_component = twoMatrix_product_sumation(gradient_matrix, time_derivative, 0)
    b2_component = twoMatrix_product_sumation(gradient_matrix, time_derivative, 1)
    b_vector = np.array([b1_component, b2_component])
    return b_vector

def applyCramerRule(A_matrix, B_vector):
    detA = np.linalg.det(A_matrix)
    dX = (B_vector[0] * A_matrix[1,1] - A_matrix[0,1] * B_vector[1])/detA
    dY = (A_matrix[0,0] * B_vector[1] - A_matrix[1,0] * B_vector[0])/detA
    d_vector = (dY,dX)
    return d_vector


#Aqui se plantea el sistema de ecuaciones del que se despeja la velocidad 
#La solución obtenida aqui se usara como candidato inicial para el algoritmo de optimización 
def solveForVelocity(reference_frame, new_frame):
    gradient_matrix = computeGradients(reference_frame, new_frame)
    print('\ndone with gradient matrix...')
    time_derivative = computeTimeDerivativeMatrix(reference_frame, new_frame)
    print('done with time deriv...')
    #se calculan aqui los compontes i,j de la matriz A del sistema Ab = d
    A_matrix = ensemble_matrix_A(gradient_matrix)
    print('done with A..')
    #Calcular el vector de terminos independientes
    B_vector = compute_vector_b(gradient_matrix, time_derivative)
    print('done with B...')
    #Calcular d = (dx,dy)
    #Por Cramer:
    motionVector = applyCramerRule(A_matrix, B_vector)
    print('done with cramer\n\n')
    return motionVector



def generateNextFrame(from_y, to_y, from_x, to_x, motionX, motionY):
    newFrame = cv.imread('black.jpeg',0)
   # newFrame = cv.normalize(newFrame.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    newFrame = im2double(newFrame)
    homogenize(newFrame, 0.50)
    paintRectangle(newFrame, from_y + motionY, to_y + motionY, from_x + motionX, to_x + motionX, 1.0)
    return newFrame


def printMatrix(matrix):
    for i in range(matrix.shape[0]):
        sys.stdout.write('\n')
        sys.stdout.flush()
        for j in range(matrix.shape[1]):
            sys.stdout.write(str(matrix[i,j]))
            sys.stdout.flush()
            sys.stdout.write(str(' '))
            sys.stdout.flush()
    print('\n')


def applyDisplacement(ref_img, displacementX, displacementY):
    newFrame = ref_img
    interpolation = interpolation2d(ref_img)
    for i in range(2, newFrame.shape[0] - 2):
        for j in range(2, newFrame.shape[1] - 2):
            x_new = i + displacementX;
            y_new = j + displacementY;
            newFrame[i,j] = interpolation(y_new,x_new)

    return newFrame


def paintRectangle(img, from_y, to_y, from_x, to_x, color):
    for i in range(from_x, to_x):
        for j in range(from_y, to_y):
            img[i,j] = color

def homogenize(img, color):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = color


def computeResiduals(reference_frame, actual_frame):
    residual_image = np.zeros((reference_frame.shape[0], reference_frame.shape[1]), float)
    for i in range(reference_frame.shape[0]):
        for j in range(reference_frame.shape[1]):
            residual_image[i,j] = abs(actual_frame[i,j] - reference_frame[i,j])
    return residual_image



def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out



def main():
    
    reference_frame = cv.imread('black.jpeg',0)

    #reference_frame = cv.normalize(reference_frame.astype('float'),None, 0.0, 1.0, cv.NORM_MINMAX)     
    reference_frame = im2double(reference_frame)
    homogenize(reference_frame, 0.55)
    paintRectangle(reference_frame,50, 100,70,120, 1.0)
    desplazamiento_x = abs(int(input('introduzca un desplazamiento sobre el eje X (solo naturales)\n')))
    desplazamiento_y = abs(int(input('introduzca un desplazamiento sobre el eje Y (solo naturales)\n')))
    
    new_frame = None
    if (desplazamiento_x + 120 <= reference_frame.shape[0] and desplazamiento_x + 120 >= 0) and (desplazamiento_y + 100 <= reference_frame.shape[1] and desplazamiento_y + 100 >= 0):
        new_frame = generateNextFrame(50, 100,70,120, desplazamiento_y, desplazamiento_x)
    else:
        print('displacement is out of boundaries')
        exit(0)


    reference_frame_ = cv.normalize(reference_frame.astype('float'), None, 0, 255, cv.NORM_MINMAX)
    new_frame_ = cv.normalize(new_frame.astype('float'), None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite('ref.png', reference_frame_)
    cv.imwrite('moving.png', new_frame_)

    fast_solution = input('\n¿Utilizar la solución de la ecuación de movimiento restringido como candidato inicial (metodo rapido) o empezar el metodo de gradiente de descenso con (0,0)?\n(Contestar con Y o N)\n\n')
    initial_solution = None
    if fast_solution == 'Y' or fast_solution == 'y':
        initial_solution = solveForVelocity(reference_frame, new_frame)
        print('Initial Solution = ', initial_solution, '\n')
    else:
        initial_solution = (0,0)
   

    epochs = 200
    alpha = 0.08

    motion_vector = computeGradientDescent(epochs, .00001, alpha, reference_frame, new_frame, initial_solution[0], initial_solution[1])
    print('Optimal motion = ', motion_vector)
    displaced_image = applyDisplacement(reference_frame, motion_vector[0], motion_vector[1])
    residual_image = computeResiduals(displaced_image, new_frame)
    residual_image = cv.normalize(residual_image.astype('float'), None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite( 'imagenResidual.png', residual_image,)
    displaced_image = cv.normalize(displaced_image.astype('float'), None, 0, 255, cv.NORM_MINMAX)
    cv.imwrite('imagenReferencia_desplazada.png', displaced_image)
    print('\n(Residuales y imagen de referencia desplazada se encuentran escritas en la carpeta en donde se ubica el file)')

main()