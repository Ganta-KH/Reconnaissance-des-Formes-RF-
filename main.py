from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QMessageBox
from PyQt5.uic import loadUi
import time
import shutil

import tools
from PIL import Image
import numpy as np
import imageViewr
import copy
import os

class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        self.imagee = imageViewr.QImageViewer()   

        QMainWindow.__init__(self)
        loadUi("assets/MainWindows.ui",self)
        self.setWindowTitle("RF")
        self.imagename = None
        self.pixel = None
        self.binaryImg = None
        self.morph = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # filtre pour dilatation, érosion, ouverture, fermeture et détection de contour 
        
        """ faire l'action en cliquant sur le bouton  """
        self.addimageB.clicked.connect(self.addimageToview)
        self.lumB.clicked.connect(self.changBrightness)
        self.ConvolutionChoice.currentIndexChanged.connect(self.convolution)
        self.resizeB.clicked.connect(self.resizeImage)
        self.rotateB.clicked.connect(self.rotatingimage)
        self.binaryB.clicked.connect(self.binary)
        self.dilationB.clicked.connect(self.dilation)
        self.erosionB.clicked.connect(self.erosion)
        self.openingB.clicked.connect(self.opening)
        self.closingB.clicked.connect(self.closing)
        self.binaryB2.clicked.connect(self.binary)
        self.ContourD_B.clicked.connect(self.contourDetection)
        self.C_Detection.clicked.connect(self.charactersDetection)
        self.save_img.clicked.connect(self.saveImg)
        

    def addimageToview(self):
        try:
            imageFile = QFileDialog.getOpenFileName(None, "Open image", os.getcwd()+"/Images", "Image Files (*.png *.jpg *.bmp *.jpeg *.png)")
            self.imagename = str(imageFile[0])
            print(self.imagename)
            self.imagee.loadimage(self.imagename)
            self.showimage()
            self.imagee.show()
            self.pixel = tools.getImage(self.imagename)
        except:
            pass
    
    def showimage(self):
        self.photoL.setPixmap(QtGui.QPixmap(self.imagename))
    def showimag(self, imgName):
        self.photoL.setPixmap(QtGui.QPixmap(imgName))
        self.imagee.loadimage(imgName)

    def changBrightness(self):
        try:
            newValue = self.brightnessS.value() # obtenir la valeur de luminosité 
            alpha, beta = tools.alpha_beta(newValue)  # obtenir l'alpha (a) et la bêta (b) 

            if alpha == 1: self.showimage()  # si 1 montre l'image originale 
            else:
                start_time = time.time()
                
                newImage = tools.brightness(self.pixel, alpha, beta)  # fonctionnement de la luminosité 
                
                Image.fromarray((newImage).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")

                print("--- %s seconds ---" % (time.time() - start_time))
        except:
            pass
    
    def convolution(self):
        try:
            Value = self.ConvolutionChoice.currentIndex()
            if Value == 0: self.showimage()
            else:
                start_time = time.time()

                p = copy.deepcopy(self.pixel)
                mask, diveur, taille = tools.convolutionActivator(Value) # retourne le masque de type et sa longueur 

                newImage = tools.convolution(self.pixel, mask.tolist(), taille, diveur) # opération de convolution 
                self.pixel = copy.deepcopy(p)

                a = np.asarray(newImage)
                Image.fromarray((a).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")

                print("--- %s seconds ---" % (time.time() - start_time))
        except:
            pass

    def resizeImage(self):
        try:
            start_time = time.time()

            fact = int(self.resizeS.value())
            newImage = tools.resize(self.imagename,fact)
            Image.fromarray((newImage).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
            self.showimag("Images/Saved/image.png")

            print("--- %s seconds ---" % (time.time() - start_time))
        except:
            pass

    def rotatingimage(self):
        try:
            angin = self.angleT.text()
            if angin.strip().isdigit():
                angle = int(angin)
                newImage = tools.rotatingImage(self.imagename,angle)
                Image.fromarray((newImage).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")
            else:
                msg = QMessageBox()
                msg.setText("please enter a number!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass

    def binary(self):
        try:
            self.binaryImg = tools.binarizeImg(self.imagename)  # changer l'image en binaire 
            Image.fromarray((self.binaryImg).astype(np.uint8)).save("img_Binary.png",format="png")  # enregistrer l'image 
            self.showimag("img_Binary.png")
            self.binaryImg = tools.getImage("img_Binary.png")
        except:
            pass

    def dilation(self):
        try:
            try:
                start_time = time.time()

                value = self.morphTrans.value()  #valeur de dilatation
                b = copy.deepcopy(self.binaryImg)
                newImg = tools.dilatation(self.binaryImg, self.morph, value)  # opération de dilatation
                a = np.asarray(newImg)
                Image.fromarray((a).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")
                self.binaryImg = copy.deepcopy(b)

                print("--- %s seconds ---" % (time.time() - start_time))
                
            except:
                msg = QMessageBox()
                msg.setText("the image not in binary!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass
            
    def erosion(self):
        try:
            try:
                start_time = time.time()

                value = self.morphTrans.value()   # valeur d'erosion
                b = copy.deepcopy(self.binaryImg)
                newImg = tools.erosion(self.binaryImg, self.morph, value)  # opération d'erosion
                a = np.asarray(newImg)
                Image.fromarray((a).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")
                self.binaryImg = copy.deepcopy(b)

                print("--- %s seconds ---" % (time.time() - start_time))
            except:
                msg = QMessageBox()
                msg.setText("the image not in binary!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass

    def opening(self):
        try:
            try:
                start_time = time.time()

                value = self.morphTrans.value()   #  valeur d'ouverture 
                b = copy.deepcopy(self.binaryImg)
                newImg = tools.opening(self.binaryImg, self.morph, value)  # opération d'ouverture 
                a = np.asarray(newImg)
                Image.fromarray((a).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")
                self.binaryImg = copy.deepcopy(b)

                print("--- %s seconds ---" % (time.time() - start_time))
            except:
                msg = QMessageBox()
                msg.setText("the image not in binary!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass

    def closing(self):
        try:
            try:
                start_time = time.time()

                value = self.morphTrans.value()  # valeur de fermeture
                b = copy.deepcopy(self.binaryImg)
                newImg = tools.closing(self.binaryImg, self.morph, value)   # opération de fermeture 
                a = np.asarray(newImg)
                Image.fromarray((a).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")
                self.binaryImg = copy.deepcopy(b)

                print("--- %s seconds ---" % (time.time() - start_time))
            except:
                msg = QMessageBox()
                msg.setText("the image not in binary!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass

    def contourDetection(self):
        try:
            try:
                start_time = time.time()

                value = self.ContourD_V.value()
                b = copy.deepcopy(self.binaryImg)
                newImg = tools.contourDetection(self.binaryImg, self.morph, value)  # opération de détection de contour 
                a = np.asarray(newImg)
                Image.fromarray((a).astype(np.uint8)).save("Images/Saved/image.png",format="png")  # enregistrer l'image 
                self.showimag("Images/Saved/image.png")
                self.binaryImg = copy.deepcopy(b)

                print("--- %s seconds ---" % (time.time() - start_time))
                
            except:
                msg = QMessageBox()
                msg.setText("the image not in binary!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass

    def deleteFiles(self):  # supprimer des fichiers dans un dossier 
        folder = 'Images/Char'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def charactersDetection(self):  # l'image doit être noire avec du texte blanc 
        try:
            try:
                start_time = time.time()
                
                self.binary()

                horizantel_separated_image = tools.markTheCharactersHorihorizantel(self.binaryImg)  # séparer les textes horizontaux de l'image 
                char_pos = tools.get_Characters_Pos_horizantel(horizantel_separated_image) # obtenir les positions des lignes de texte 
                lines = tools.get_the_charactes_mat_horizantel(horizantel_separated_image, char_pos) # renvoie chaque ligne de texte dans la matrice (image) 

                self.deleteFiles()  # supprimer les anciennes images (caractères) de la dernière opération 

                Image.fromarray((horizantel_separated_image).astype(np.uint8)).save("Images/Charhorizantel_separated.png",format="png")  # enregistrer l'image 
                image = Image.open("Images/Charhorizantel_separated.png")
                image.show()  # montrer l'image horizontalement séparée 
                for i in range(0,len(lines), 2):
                    Image.fromarray((lines[i]).astype(np.uint8)).save("Images/Chartext_line"+str(i)+".png",format="png")
                    image = Image.open("Images/Chartext_line"+str(i)+".png")
                    image.show()  #montrer chaque ligne de texte qu'il trouve 

                for i in range(0 ,len(lines), 2):
                    a ,d = tools.characters_edge(lines[i].tolist()) # renvoie la hauteur d'un caractère d'où il commence a (en haut) et où il se termine d (en bas) 
                    marked_image = tools.markTheCharacters(lines[i], d ,a)  # séparer le caractère avec une ligne verte 
                    characters_pos = tools.get_Characters_Pos(marked_image)    # obtenir les positions de tous les caractère
                    characters = tools.get_the_charactes_mat(np.array(lines[i]), characters_pos, d, a)  # renvoie chaque matrice d'image de caractère 

                    Image.fromarray((marked_image).astype(np.uint8)).save("Images/Charchar_Detected"+str(i)+".png",format="png")  # enregistrer l'image 
                    image = Image.open("Images/Charchar_Detected"+str(i)+".png")
                    image.show()  # afficher les lignes de texte avec des caractères séparés 

                    for j in range(0, len(characters), 2):
                        Image.fromarray((characters[j]).astype(np.uint8)).save("Images/Charcharacters"+str(j)+".png",format="png")  # enregistrer les caractères en tant qu'images 
                        image = Image.open("Images/Charcharacters"+str(j)+".png")
                        image.show()  # afficher chaque caractère 
                
                print("--- %s seconds ---" % (time.time() - start_time))
            except:
                msg = QMessageBox()
                msg.setText("the image not in binary!")
                msg.setWindowTitle("Error")
                msg.exec_()
        except:
            pass

    def saveImg(self):
        if self.imagename != None:
            new_name = str(np.random.randint(999999))+".png"
            os.rename(r'Images/Saved/image.png', str("Images/Saved/")+new_name)
            self.imagename = str("Images/Saved/")+new_name
            print(self.imagename)
            self.pixel = tools.getImage(self.imagename)
            


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
