from PIL import Image
import numpy as np
import math
import cv2
from copy import deepcopy

def getImage(image_path):    # obtenir les pixels de la matrice 
    im = Image.open(image_path)   # ouvrir l'image 
    im_rgb = im.convert("RGB")   # le convertir en RGB 
    pixel = np.array(im_rgb)   # RGB matrice
    pixel = pixel.tolist()
    return pixel    # return la matrice

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def alpha_beta(value):    # alpha et beta valeur
    """
    Diminution de contraste : a < 1 et b > 0
    Augmentation de contraste : a > 1 et b < 0
    """
                      # alpha (a), beta (b)
    if value == 0: return 0.04, 50
    if value == 1: return 0.1, 50
    if value == 2: return 0.2, 50
    if value == 3: return 0.4, 50
    if value == 4: return 0.67, 50
    if value == 5: return 1, 0  # image originale 
    if value == 6: return 1.5, -20
    if value == 7: return 2.5, -50
    if value == 8: return 5, -50
    if value == 9: return 10, -50
    if value == 10: return 25, -50

def brightness(image, alpha, beta):
    image = np.array(image)
    image = alpha * image + beta    # changer la luminosité avec alpha et bêta de toute la matrice d'image
    image = np.where(image < 255, image, 255) # tous les pixels supérieurs à 255 sont égaux à 255 
    image = np.where(image > 0, image, 0) # tous les pixels inférieurs à 0 égaux à 0 
    return image    # le nouveau résultat 
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def addZeros(image, k):  # ajouter un cadre de pixels ( [0, 0, 0] ) à la matrice pour faire l'opération morphologique 
    L = len(image[0])
    for _ in range(k):
        image.insert( 0, [(0, 0, 0) for _ in range( L )] ) # ajouter par dessus l'image 
        image.append( [(0, 0, 0) for _ in range( L ) ] )    # ajouter en bas de l'image 
    for img in image:
        for _ in range(k):
            img.insert(0, (0, 0, 0)) # ajouter des pixels à gauche 
            img.append((0, 0, 0))   # ajouter des pixels à droit
    return image

def convSum(var):
    if var > 255: return 255
    elif var < 0: return 0
    return var

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def convolutionActivator(value):
    #if value == 1: return np.array(([0, 0, 0], [0, 1, 0], [0, 0, 0])), 1, 1   #identity
    if value == 1: return np.array(([0, 1, 0], [1, 4, 1], [0, 1, 0])), 1/8, 1   #identity
    if value == 2: return np.array(([1, 0, -1], [0, 0, 0], [-1, 0, 1])), 1, 1   #edge detection 1
    if value == 3: return np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0])), 1, 1   #edge detection 2
    if value == 4: return np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1])), 1, 1   #edge detection 3
    if value == 5: return np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0])), 1, 1   # sharpen
    if value == 6: return np.ones((3,3)), 1/9, 1   # Box blur 3 x 3
    if value == 7: return np.ones((5,5)), 1/25, 2   # Box blur 5 x 5
    if value == 8: return np.ones((7,7)), 1/49, 3   # Box blur 7 x 7
    if value == 9: return np.array(([1 ,2 , 1], [2, 4, 2], [1 ,2 ,1])), 1/16, 1   #1/16 Gaussian blur 3 x 3
    if value == 10: return np.array(([1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1])), 1/256, 2   # 1/256 Gaussian blur 5 × 5
    if value == 11: return np.array(([1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1])), -1/256, 2   # Unsharp masking 5 x 5 -1/256
    
def convolutionCalcule(mat, mask, k, posx, posy, diveur): # calculat one pixel with the mask
    h1, r, g , b = 0, 0, 0, 0
    for f in range(k*-1, k+1):
        h2 = 0
        for repeat in range(k*-1, k+1):
            r += mat[posx+f][posy+repeat][0] * mask[h1][h2]   # Calculez la sum( red coleur * mask )
            g += mat[posx+f][posy+repeat][1] * mask[h1][h2]   # Calculez la sum( green coleur * mask )
            b += mat[posx+f][posy+repeat][2] * mask[h1][h2]   # Calculez la sum( blue coleur * mask )
            h2 += 1
        h1 += 1
    return (convSum(r*diveur), convSum(g*diveur), convSum(b*diveur)) # retourner le nouveau rgb du pixel sur lequel nous avons travaillé 

def convolution(mat, mask, k, diveur):
    newImg = []
    mat = addZeros(mat, k)   # ajouter un cadre de pixels ( [0, 0, 0] ) à la matrice
    for i in range( k, len(mat) - k ):
        newImg.append( [] )
        for j in range( k, len(mat[i]) - k):
            newImg[i-k].append( convolutionCalcule(mat, mask, k, i, j, diveur) ) # calculer chaque pixel avec le masque (la fonction au-dessus de celui-ci) 
    return newImg  # retourner le nouveau résultat

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def resize(imagename,factor):
    im1 = Image.open(imagename)
    image_array = np.array(im1) #cela transforme l'image en un tableau afin que nous puissions obtenir la largeur et la hauteur comme les 2 lignes suivantes 
    W = image_array.shape[1]
    H = image_array.shape[0]

    if factor>5: #ici quand le facteur > 5 ça veut dire qu'on va agrandir l'image 
        factor -= 4 #on met -4 car 5 est l'image d'origine signifie 5-4 =1 et 6-4= 2 signifie la double taille et ainsi de suite 
        newW = W*factor #la nouvelle largeur de l'image serait *facteur 
        newH = H*factor #la nouvelle hauteur de l'image serait *facteur 
        newImage = np.zeros((newH,newW,3)) #nous remplissons un tableau de nouvelles hauteur et largeur avec 3 zéros chaque cas signifie RVB 0 

        for col in range(newW):
            for row in range(newH):
                p = image_array[row//factor][col//factor] #nous parcourons tous les cas du tableau et nous le remplissons avec le RVB de l'ancienne image mais plus d'une fois 
                newImage[row][col] = p 
    else: #ici quand le facteur < 5 ça veut dire qu'on va rétrécir l'image 
        factor = 6 - factor # 6- facteur car 5 est original ... 6-5 = 1 et 6 - 4 = 2 signifie la moitié de l'image 
        newW = W//factor#la nouvelle largeur de l'image serait /facteur 
        newH = H//factor#la nouvelle hauteur de l'image serait /facteur 
        newImage = np.zeros((newH,newW,3)) #nous remplissons un tableau de nouvelles hauteur et largeur avec 3 zéros chaque cas signifie RVB 0 

        for col in range(newW):
            for row in range(newH):
                p = image_array[row*factor-factor][col*factor-factor]#nous parcourons tous les cas du tableau et nous le remplissons avec le RVB de l'ancienne image mais moins que le nombre de pixels d'origine  
                newImage[row][col] = p
    return newImage

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def rotatingImage(imageName,angle):
    image = np.array(Image.open(imageName))#cela transforme l'image en un tableau afin que nous puissions obtenir la largeur et la hauteur

    angle=math.radians(angle) #nous faisons l'entrée de l'utilisateur comme un angle pour que python le comprenne comme ça                             
    cosine=math.cos(angle)# on prend le cos de l'angle 
    sine=math.sin(angle)# on prend le sin de l'angle 
    height=image.shape[0]                                 
    width=image.shape[1]      

    new_height  = round(abs(height*cosine)+abs(width*sine))+1 # nous faisons des calculs ici pour obtenir la nouvelle hauteur d'image 
    new_width  = round(abs(width*cosine)+abs(height*sine))+1 # nous faisons quelques calculs ici pour obtenir la nouvelle largeur de l'image 

    newimage=np.zeros((new_height,new_width,image.shape[2])) #nous remplissons la nouvelle image avec des zéros ... image.shape[2] = 3 cela signifie tuple RVB de l'image et nous les remplissons de zéros 
    
    # Trouver le centre de l'image autour duquel nous devons faire pivoter l'image 
    original_centre_height   = round(((height+1)/2)-1)  
    original_centre_width    = round(((width+1)/2)-1)  
    
    # Trouver le centre de la nouvelle image qui sera obtenue 
    new_centre_height= round(((new_height+1)/2)-1)        
    new_centre_width= round(((new_width+1)/2)-1)       

    for i in range(height):
        for j in range(width):
            #coordonnées du pixel par rapport au centre de l'image originale 
            y=height-1-i-original_centre_height                   
            x=width-1-j-original_centre_width                
            #coordonnée du pixel par rapport à l'image pivotée 
            new_y=round(-x*sine+y*cosine)
            new_x=round(x*cosine+y*sine)
            #puisque l'image sera tournée, le centre changera aussi, donc pour s'adapter à cela, nous devrons changer new_x et new_y par rapport au nouveau centre 
            new_y=new_centre_height-new_y
            new_x=new_centre_width-new_x
            # ajouter if check pour éviter toute erreur dans le traitement 
            if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                newimage[new_y,new_x,:]=image[i,j,:]   #écriture des pixels vers la nouvelle destination dans l'image de sortie                        

    return newimage

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def readImg(image):  # ouvrir l'image 
    image = cv2.imread(image, 0)
    return image

def convertBinary(image, val):  #  convertir en binaire 
    bin = np.where((image <= val), image, 255) # tous les pixels supérieurs à 255 sont égaux à 255 
    final_bin = np.where((bin > val), bin, 0) # tous les pixels inférieurs à 0 sont égaux à 0 
    return final_bin

def binarizeImg(image, val=127):  # ouvrir et convertir l'image en binaire  
    image = readImg(image)
    image_b = convertBinary(image, val)
    return image_b
    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def dilitationCalcule(mat, filter, k, posx, posy, center):  # vérifier les pixels environnants de pixel[posx,posy] avec un filtre  and operation dilatation
    h1 = 0
    for f in range(k*-1, k+1, k):
        h2 = 0
        for repeat in range(k*-1, k+1, k):
            if filter[h1][h2] == 1: 
                rgb = mat[posx+f][posy+repeat][0]
                if rgb > center:     # si l'un des pixels environnants n'est pas le même, changez celui-ci en couleur blanche 
                    return [255, 255, 255]
            h2 += 1
        h1 += 1
    return [0, 0, 0] # si tout l'entourage est noir, gardez-le noir 

def dilatation(image, filter, k):
    image = addZeros(image, k)  # ajouter un cadre de pixels ( [0, 0, 0] ) à la matrice
    newImg = []
    for i in range( k, len(image) - k ):
        newImg.append( [] )
        for j in range( k, len(image[i]) - k):
            center = image[i][j][0]
            if center == 255: 
                newImg[i-k].append([255, 255, 255]) # si le pixel principal est égal au blanc, gardez-le blanc 
            else:
                newImg[i-k].append( dilitationCalcule(image, filter, k, i, j, center) ) # vérifier la couleur du pixel environnant (la fonction au-dessus de celle-ci) 
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def erosionCalcule(mat, filter, k, posx, posy, center): # vérifier les pixels environnants de pixel[posx,posy] avec un filtre avec operation erosion
    h1 = 0
    for f in range(k*-1, k+1, k):
        h2 = 0
        for repeat in range(k*-1, k+1, k):
            if filter[h1][h2] == 1: 
                rgb = mat[posx+f][posy+repeat][0]
                if rgb < center:    # si l'un des pixels environnants n'est pas le même, changez le pixel en couleur noire 
                    return [0, 0, 0]
            h2 += 1
        h1 += 1
    return [255, 255, 255] # si tout ce qui l'entoure est de couleur blanche, gardez-le blanc 

def erosion(image, filter, k):
    image = addZeros(image, k)  # ajouter un cadre de pixels ( [0, 0, 0] ) à la matrice
    newImg = []
    for i in range( k, len(image) - k ):
        newImg.append( [] )
        for j in range( k, len(image[i]) - k):
            center = image[i][j][0]
            if center == 0: 
                newImg[i-k].append([0, 0, 0])  # si le pixel principal est égal au noir, gardez-le noir 
            else:
                newImg[i-k].append( erosionCalcule(image, filter, k, i, j, center) ) # vérifier la couleur du pixel environnant (la fonction au-dessus de celle-ci) 
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def opening(image, filter, k):
    erose = erosion(image, filter, k)
    newImg = dilatation(erose, filter, k)
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def closing(image, filter, k):
    dilation = dilatation(image, filter, k)
    newImg = erosion(dilation, filter, k)
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def contourDetection(image, filter, k):
    image = addZeros(image, k)  # ajouter un cadre de pixels ( [0, 0, 0] ) à la matrice
    newImg = []
    for i in range( k, len(image) - k ):
        newImg.append( [] )
        for j in range( k, len(image[i]) - k):
            center = image[i][j][0]  # une variable uniquement pour aider à la vérification 
            newImg[i-k].append( dilitationCalcule(image, filter, k, i, j, center) ) # obtenir le coutour à l'aide de l'opération de dilatation (la fonction dilatationCalacule) 
    return newImg

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



# l'image doit être noire avec du texte blanc 
# l'image doit être noire avec du texte blanc 
# l'image doit être noire avec du texte blanc 

def markTheCharactersHorihorizantel(image):    # séparer les textes horizontaux de l'image 
    new_image = deepcopy(np.array(image))
    check , k = [0, 0, 0], [0, 0, 0]
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] == [255,255,255]:   # vérifie si la couleur passe au blanc, puis il trouve une ligne de texte 
                k = [255, 255, 255]
                break
            k = [0, 0, 0]
        if (k != check):
            new_image[i, 0: ] = [0, 255, 0]   # tracer une ligne verte horizontale dans l'image où l'on trouve du texte 
            check = k.copy()
    return new_image

def get_Characters_Pos_horizantel(image):  # obtenir les positions des lignes de texte 
    characters = []
    for i in range(len(image[0])):
        for j in range(len(image)):
            if (image[j][i] == [0,255,0]).all():   # si le pixel est vert retourne sa position 
                characters.append(j)
        if len(characters) != 0:  # s'il obtient toutes les positions des lignes de texte, retourner chaque position de ligne dans l'image 
            break
    return characters

def get_the_charactes_mat_horizantel(image, characters):  # renvoie chaque ligne de texte dans la matrice (image) 
    pos = 0
    result = []
    while pos+1 != len(characters):
        result.append( [] )
        result[pos] = image[characters[pos]+1: characters[pos+1], 0: ]   # toutes les deux lignes vertes renvoient la matrice de lignes de texte (image) 
        pos += 1
    return result

################################################################################

def characters_edge(image): # renvoie la hauteur d'un caractère d'où il commence a (en haut) et où il se termine d (en bas) 
    a, d = 0, 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if a == 0 and (image[i][j] == [255, 255, 255]): # où ça commence la hauteur pour le premier caractère 
                a = [i, j]
            if d == 0 and (image[-i][j] == [255, 255, 255]): # où ça s'arrête le premier caractères
                d = [-i, j]
        if (a and d) != 0:
            break
    for i in range(len(image)):
        for j in range(len(image[i])):
            if a[0] < i and (image[i][j] == [255, 255, 255]):  # où ils commencent la hauteur pour tout les caractères 
                a[0] = i
            if d[0] > -i and (image[-i][j] == [255, 255, 255]): # où ça s'arrête pour tout les caractères
                d[0] = -i
    return a, d

#detect characters
def markTheCharacters(image, d, a):  # séparer le caractère avec une ligne verte 
    new_image = deepcopy(image)
    check , k = [0, 0, 0], [0, 0, 0]
    for i in range(len(image[0])):
        for j in range(len(image)):
            if (image[j][i] == [255,255,255]).all():  # vérifie si la couleur passe au blanc alors il trouve un caractère
                k = [255, 255, 255]
                break
            k = [0, 0, 0]
        if (k != check):
            new_image[len(image) + d[0]-1 : a[0]+1, i] = [0, 255, 0]  # séparer les caractères avec une ligne verte 
            check = k.copy()
    return new_image

def get_Characters_Pos(image):  # obtenir les positions de tous les caractère
    characters = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if (image[i][j] == [0,255,0]).all():  # si le pixel est vert ([0, 255, 0]) retourne sa position 
                characters.append(j)
        if len(characters) != 0:
            break
    return characters

def get_the_charactes_mat(image, characters, d, a):   # renvoie chaque matrice d'image de caractère 
    pos = 0
    result = []
    while pos+1 != len(characters):
        result.append( [] )
        result[pos] = image[len(image) + d[0]-1 : a[0]+1, characters[pos]-1: characters[pos+1]+1] # toutes les deux lignes vertes renvoient la matrice de caractères (image) 
        pos += 1
    return result