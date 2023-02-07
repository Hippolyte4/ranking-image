
##########################
##CODAGE RANKING D'IMAGE##
##########################


#Importation des bibliothèques utilisées

import time
import PIL
import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy import ndimage, signal
from imutils import build_montages
from imutils import paths
import imutils
import argparse


# Initialisation du temps

start=time.time()

# Chemins des fichiers

path_ref = "/home/stagiare/Desktop/crops/exemple/SIGNRONDB100/ref.jpg"
path = "/home/stagiare/Desktop/crops/exemple/SIGNRONDB100/"
path_write_ref = "/home/stagiare/Desktop/crops/exemplerecrop/ref.jpg"
path_write = "/home/stagiare/Desktop/crops/exemplerecrop/"

# Def liste fichiers

fichiers = [f for f in listdir(path) if isfile(join(path, f))]
a=len(fichiers)

# Def de variables

pixels_horizontaux, pixels_verticaux = 94,94
L, B,lum, contraste, REF, DIFFABS = [],[],[],[],[],[]
CONTOUR,DIFFABS_CONT_HOR, DIFFABS_CONT_VER =[],[],[] 
compteur=0
centrage_contraste = 18
RESULTS=[]


# Définition des filtres/noyaux pour déterminer la finesse des contours :

noyau_v = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

noyau_h = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

    ####################################################################
    ##### Définition image de référence                           ######
    ##### /home/stagiare/Desktop/crops/exemple/SIGNROND10/ref.jpg ######
    ####################################################################

imgref=cv2.imread(path_ref)
imgref_resized = cv2.resize(imgref, (pixels_horizontaux, pixels_verticaux))
imgnpref0  =cv2.imread(path_write_ref)
cv2.imwrite(path_write_ref, imgref_resized)
#print(np.shape(imgnpref))
teinte_grise_ref = cv2.cvtColor(imgref_resized, cv2.COLOR_BGR2GRAY)
mean_brightness_ref = np.mean(teinte_grise_ref)
lum.append(mean_brightness_ref)
laplacian_ref = ndimage.filters.laplace(imgref_resized)
sharpness_ref = np.mean(np.abs(laplacian_ref))
np_max_ref = np.max(teinte_grise_ref)
np_min_ref = np.min(teinte_grise_ref)
res2ref = int(np_max_ref) + int(np_min_ref)
res1ref = int(np_max_ref) - int(np_min_ref)
contraste_ref=(res1ref/res2ref)
B.append(mean_brightness_ref)
B.append(sharpness_ref) 
B.append(res1ref/res2ref)
B.append("ref.jpg")

#Définition des coefficients pondérant les différents facteurs dans la moyenne finale

coeff_brightness = 10/B[0]
coeff_sharpness = 70/B[1]
coeff_contraste = 20/B[2]

REF.append(mean_brightness_ref*coeff_brightness)
REF.append(sharpness_ref*coeff_sharpness)
REF.append(contraste_ref*coeff_contraste)

resulting_image_ref_ver = cv2.filter2D(imgref_resized, -1, noyau_v)
resulting_image_ref_hor = cv2.filter2D(imgref_resized, -1, noyau_h)
CONTOUR=[(resulting_image_ref_hor,resulting_image_ref_ver,"ref.jpg")]

mean_ref, stddev_ref = cv2.meanStdDev(teinte_grise_ref)
threshold_ref = mean_ref + 2 * stddev_ref
noisy_pixels_ref = np.where(teinte_grise_ref > threshold_ref)

# Fonction pour calculer la "colorfulness"/vibrance d'une image

def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


    #########################################################

for f in fichiers:

    # Normalisation en format 94x94 pixels puis conversion de l'image en tableau numpy

    img=cv2.imread(path + f)

    img_resized = cv2.resize(img, (pixels_horizontaux, pixels_verticaux))
    cv2.imwrite(path_write+f[0:len(f)-4]+"_recroped.jpg", img_resized)
    imgnp  =cv2.imread(path_write+f[0:len(f)-4]+"_recroped.jpg")
    b=np.shape(imgnp)
    teinte_grise = cv2.cvtColor(imgnp, cv2.COLOR_BGR2GRAY)
    longueur=int(len(teinte_grise)/2)
    for i in range(len(teinte_grise)):
        for j in range(len(teinte_grise[0])):
            if i<longueur - centrage_contraste or i > longueur + centrage_contraste or j < longueur - centrage_contraste or j > longueur + centrage_contraste:       
                teinte_grise[i][j]=teinte_grise[longueur][longueur]      

    # Calcul de la luminosité moyenne
    
    mean_brightness = np.mean(teinte_grise)
    lum.append(mean_brightness)

    # Calcul de la sharpness/netteté

    laplacian = ndimage.filters.laplace(imgnp)
    sharpness = np.mean(np.abs(laplacian))
    #cv2.imwrite("/home/stagiare/Desktop/crops/exemplerecrop/" +f+"_recroped_grisée", teinte_grise)

    # Calcul du contraste

    np_max = np.max(teinte_grise)                
    np_min = np.min(teinte_grise)
    res2 = int(np_max) + int(np_min)           
    res1 = int(np_max) - int(np_min)         
    r = float(res1)/float(res2)
    contraste.append((str(r), f))

    L.append(((mean_brightness, coeff_brightness), (sharpness, coeff_sharpness), (res1/res2, coeff_contraste),f))   # liste contenant la luminosité moyenne (L[i][0]) et son coeff
                                                                                                                    # la netteté (L[i][1]) et son coeff
                                                                                                                    # le contraste et son coeff
                                                                                                                    # le nom du fichier


# Comparaison avec l'image de référence

    # Calcul de la différence absolue entre chaque pixel    

    diff = cv2.absdiff(imgref_resized, imgnp)
    result = np.sum(diff)
    DIFFABS.append((result, f))
    
    # "Finesse" des contours 
       
    resulting_image_ver = cv2.filter2D(imgnp, -1, noyau_v)
    resulting_image_hor = cv2.filter2D(imgnp, -1, noyau_h)
    CONTOUR.append((resulting_image_hor, resulting_image_ver,f))

    diff_contour_hor=cv2.absdiff(resulting_image_ref_hor,resulting_image_hor)
    result_hor = np.sum(diff_contour_hor)
    DIFFABS_CONT_HOR.append((result_hor,f))
    DIFFABS_CONT_HOR1=sorted(DIFFABS_CONT_HOR, key=lambda x: x[1])
    diff_contour_ver=cv2.absdiff(resulting_image_ref_ver,resulting_image_ver)
    result_ver = np.sum(diff_contour_ver)
    DIFFABS_CONT_VER.append((result_ver,f))
    DIFFABS_CONT_VER1=sorted(DIFFABS_CONT_VER, key=lambda x: x[1])
    

# Comparaison du bruit des images (pas utilisée car résultats aberrants)
    
    # mean, stddev = cv2.meanStdDev(teinte_grise[longueur-centrage_contraste:longueur+centrage_contraste][longueur-centrage_contraste:longueur+centrage_contraste])
    # threshold = mean + 2 * stddev
    # noisy_pixels = np.where(teinte_grise > threshold)
    # print(noisy_pixels[0])
    # print(threshold,f,len(noisy_pixels[0]))

# Comparaison de la vibrance des images

    debut, fin = longueur- centrage_contraste, longueur+centrage_contraste
    results = []
    C = image_colorfulness(imgnp[debut:fin])
    results.append((C, f))
    RESULTS.append(results[0])
    m=max(RESULTS[i][0] for i in range(len(RESULTS)))
    RESULTS1 = sorted(RESULTS, key=lambda x: x[1])

# Création de la liste de "travail"

J, COMP=[],[]

for k in range(len(L)):
   H=[L[k][j][0]*L[k][j][1] for j in range(len(L[0])-1)]        
   J.append((H,L[k][len(L[0])-1]))

############################################################################
##Critères "absolus" de l'image (pas de comparaison avec une image de ref)##
############################################################################

for j in range(len(J)):
    o = abs(J[0][0][0]-J[j][0][0])
    p = abs(J[0][0][1]-J[j][0][1])
    q = abs(J[0][0][2]-J[j][0][2])
    COMP.append(((o+p+q)/100,J[j][1]))

COMP1=sorted(COMP, key=lambda x: x[1])

# Les listes triées par nom sont indicées avec un 1

#COMP1=COMP0[0:len(J)-2]#+[(0.1821395714238287,'I.jpg')]+[(0,'ref.jpg')] #truc étrange, les 2 dernières valeurs sont échangées!
                                                                       #besoin de bidouiller un peu


#########################################
##Comparaison avec l'image de référence##
#########################################

# Les listes triées par nom sont indicées avec un 1
DIFFABS1=sorted(DIFFABS, key=lambda x: x[1])

for f in fichiers:
    DIFF=[(DIFFABS_CONT_VER1[i][0] + DIFFABS_CONT_HOR1[i][0],DIFFABS1[i][1])for i in range(len(DIFFABS1))]
    
# Liste triée par nom de fichier des différences absolues pour les contours
DIFF_CONTOUR1=sorted(DIFF, key=lambda x: x[1])


for f in fichiers:
    DIFF_ORDERED=[(DIFFABS1[i][0], DIFFABS1[i][1]) for i in range(len(DIFFABS1))]

# Liste triée par nom de fichier des différences absolues entre chaque pixels
DIFF_ORDERED1=sorted(DIFF_ORDERED, key=lambda x: x[1])

a=abs(min(DIFF_CONTOUR1[i][0] for i in range(len(DIFF))))
m1=min(DIFF_ORDERED1[i][0] for i in range(len(DIFF)))
m2=max((RESULTS1[i][0] for i in range(len(DIFF))))

# Calcul final et ranking

RANKING=[]

# On pose un calcul de telle sorte à ce que plus le score est proche de 0, 
# plus l'image est de bonne qualité

for i in range(len(DIFF_ORDERED)):
    a=10*COMP1[i][0] +(1- 0.5*RESULTS1[i][0]/m2)+ (1-0.05*DIFF_CONTOUR1[i][0]/a) + (1-DIFF_ORDERED1[i][0]/m1) 
    RANKING.append((abs(a),DIFF_CONTOUR1[i][1]))

R0=sorted(RANKING, key=lambda x: x[0])

RANKING_FINAL=[str(R0[i][1]) for i in range(len(R0))]

# for i in range(len(RANKING_FINAL)):
#     cv2.imshow("Image", "/home/stagiare/Desktop/crops/exemplerecrop/"+str(RANKING_FINAL[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Affichage des résultats

print(COMP1, len(COMP1))
print()
print(DIFF_CONTOUR1)
print()
print(DIFF_ORDERED1, len(DIFF_ORDERED1))
print()
print(RESULTS1, len(RESULTS1))
print()
print(a,m1,m2)
print()
print(RANKING)
print()
print(RANKING_FINAL)
print()
print("Temps d'execution :", int(1000*(time.time() - start)), "millisecondes")


# Ce qui renvoie :

# [(0.14538568484243394, 'A.jpg'), (0.10935337034890531, 'B.jpg'), (0.30420057036242015, 'C.jpg'), (0.33614211455393894, 'D.jpg'), (0.20963619175593873, 'E.jpg'), (0.18215905905645166, 'F.jpg'), (0.05231740472201185, 'G.jpg'), (0.2122618376293177, 'H.jpg'), (0.1821395714238287, 'I.jpg'), (0, 'ref.jpg')] 10

# [(2061609, 'A.jpg'), (1946511, 'B.jpg'), (1990875, 'C.jpg'), (1865478, 'D.jpg'), (1954748, 'E.jpg'), (1912955, 'F.jpg'), (2077728, 'G.jpg'), (2210594, 'H.jpg'), (1739532, 'I.jpg'), (0, 'ref.jpg')]

# [(1582503, 'A.jpg'), (1515104, 'B.jpg'), (1666936, 'C.jpg'), (1964416, 'D.jpg'), (1634830, 'E.jpg'), (1336043, 'F.jpg'), (1373016, 'G.jpg'), (1334745, 'H.jpg'), (1418373, 'I.jpg'), (0, 'ref.jpg')] 10

# [(24.32759067658958, 'A.jpg'), (18.981900537359436, 'B.jpg'), (43.05586649881294, 'C.jpg'), (19.391381701172715, 'D.jpg'), (33.06160040542959, 'E.jpg'), (32.22790310807157, 'F.jpg'), (30.428990567556525, 'G.jpg'), (50.756899793156954, 'H.jpg'), (22.11305585742446, 'I.jpg'), (81.54462230305009, 'ref.jpg')] 10
# 2210594 1964416 81.54462230305009

# ['ref.jpg', 'A.jpg', 'C.jpg', 'E.jpg', 'G.jpg', 'I.jpg', 'H.jpg', 'F.jpg', 'D.jpg', 'B.jpg']

# Temps d'execution : 67 millisecondes

