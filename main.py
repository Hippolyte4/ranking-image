#Importation des bibliothèques utilisées
# Codage pour ranking crops

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

start=time.time()

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

fichiers = [f for f in listdir("/home/stagiare/Desktop/crops/exemple/SIGNRONDB10") if isfile(join("/home/stagiare/Desktop/crops/exemple/SIGNRONDB10", f))]
a=len(fichiers)
pixels_horizontaux, pixels_verticaux = 94,94
L, B,lum, contraste, REF, DIFFABS = [],[],[],[],[],[]
CONTOUR,DIFFABS_CONT_HOR, DIFFABS_CONT_VER =[],[],[] 
compteur=0
centrage_contraste = 2


# Définition des filtres/noyaux :
noyau_v = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

noyau_h = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])


    #########################################################
    ##### Définition image de référence                ######
    ##### /home/stagiare/Desktop/crops/exemple/ref.jpg ######
    #########################################################

imgref=cv2.imread("/home/stagiare/Desktop/crops/exemple/ref.jpg")
imgref_resized = cv2.resize(imgref, (pixels_horizontaux, pixels_verticaux))
cv2.imwrite("/home/stagiare/Desktop/crops/exemplerecrop/ref.jpg", imgref_resized)
imgnpref  =cv2.imread("/home/stagiare/Desktop/crops/exemplerecrop/ref.jpg")
#print(np.shape(imgnpref))
teinte_grise_ref = cv2.cvtColor(imgnpref, cv2.COLOR_BGR2GRAY)
mean_brightness_ref = np.mean(teinte_grise_ref)
lum.append(mean_brightness_ref)
laplacian_ref = ndimage.filters.laplace(imgnpref)
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

coeff_brightness = 1/B[0]
coeff_sharpness = 1/B[1]
coeff_contraste = 1/B[2]

REF.append(mean_brightness_ref*coeff_brightness)
REF.append(sharpness_ref*coeff_sharpness)
REF.append(contraste_ref*coeff_contraste)

resulting_image_ref_ver = cv2.filter2D(imgnpref, -1, noyau_v)
resulting_image_ref_hor = cv2.filter2D(imgnpref, -1, noyau_h)
CONTOUR=[(resulting_image_ref_hor,resulting_image_ref_ver,"ref.jpg")]

mean_ref, stddev_ref = cv2.meanStdDev(teinte_grise_ref)
threshold_ref = mean_ref + 2 * stddev_ref
noisy_pixels_ref = np.where(teinte_grise_ref > threshold_ref)
cv2.waitKey(0)
cv2.destroyAllWindows()


    #########################################################

for f in fichiers:

# Normalisation en format 94x94 pixels puis conversion de l'image en tableau numpy

    img=cv2.imread("/home/stagiare/Desktop/crops/exemple/SIGNRONDB10/" + f)

    img_resized = cv2.resize(img, (pixels_horizontaux, pixels_verticaux))
    cv2.imwrite("/home/stagiare/Desktop/crops/exemplerecrop/"+f[0:len(f)-4]+"_recroped.jpg", img_resized)
    imgnp  =cv2.imread("/home/stagiare/Desktop/crops/exemplerecrop/"+f[0:len(f)-4]+"_recroped.jpg")
    b=np.shape(imgnp)
    teinte_grise = cv2.cvtColor(imgnp, cv2.COLOR_BGR2GRAY)
    for i in range(len(teinte_grise)):
        for j in range(len(teinte_grise[0])):
            if i<int(len(teinte_grise)/2) - centrage_contraste or i > int(len(teinte_grise)/2) + centrage_contraste or j < int(len(teinte_grise)/2) - centrage_contraste or j > int(len(teinte_grise)/2) + centrage_contraste:       
                teinte_grise[i][j]=teinte_grise[int(len(teinte_grise)/2)][int(len(teinte_grise)/2)]


    #print(np.shape(teinte_grise))       

# Calcul de la luminosité moyenne
    
    mean_brightness = np.mean(teinte_grise)
    lum.append(mean_brightness)

# Calcul de la sharpness/netteté

    laplacian = ndimage.filters.laplace(imgnp)
    sharpness = np.mean(np.abs(laplacian))
#    cv2.imwrite("/home/stagiare/Desktop/crops/exemplerecrop/" +f+"_recroped_grisée", teinte_grise)

# Calcul du contraste

    np_max = np.max(teinte_grise)                
    np_min = np.min(teinte_grise)
    res2 = int(np_max) + int(np_min)           
    res1 = int(np_max) - int(np_min)     
    
    r = float(res1)/float(res2)
    
    #print("res: " + str(r), f)
    #print()
    contraste.append((str(r), f))

    L.append(((mean_brightness, coeff_brightness), (sharpness, coeff_sharpness), (res1/res2, coeff_contraste),f))   # liste contenant la luminosité moyenne (L[i][0]) et son coeff
                                                                                                                    # la netteté (L[i][1]) et son coeff
                                                                                                                    # le contraste et son coeff
                                                                                                                    # le nom du fichier


# Calcul de la différence absolue entre chaque pixel    

    diff = cv2.absdiff(imgnpref, imgnp)
    result = np.sum(diff)
    DIFFABS.append((result, f))
    
# Affinage des contours 
       
    resulting_image_ver = cv2.filter2D(imgnp, -1, noyau_v)
    resulting_image_hor = cv2.filter2D(imgnp, -1, noyau_h)
    CONTOUR.append((resulting_image_hor, resulting_image_ver,f))

    diff_contour_hor=cv2.absdiff(resulting_image_ref_hor,resulting_image_hor)
    result_hor = np.sum(diff_contour_hor)
    DIFFABS_CONT_HOR.append((result_hor,f))
    diff_contour_ver=cv2.absdiff(resulting_image_ref_ver,resulting_image_ver)
    result_ver = np.sum(diff_contour_ver)
    DIFFABS_CONT_VER.append((result_ver,f))

#print(CONTOUR)
#print(DIFFABS)

# Comparaison du bruit des images
    
    mean, stddev = cv2.meanStdDev(teinte_grise)
    threshold = mean + 2 * stddev
    noisy_pixels = np.where(teinte_grise > threshold)

    #print((noisy_pixels, len(noisy_pixels)))

# Comparaison de la vibrance des images

# https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/

    results = []

    C = image_colorfulness(image)

    # add the image and colorfulness metric to the results list
    results.append((image, C))
    results = sorted(results, key=lambda x: x[1], reverse=True)


# Création de la liste de "travail"

J=[(REF, "ref.jpg")]

for k in range(len(L)):
   H=[L[k][j][0]*L[k][j][1] for j in range(len(L[0])-1)]        
   J.append((H,L[k][len(L[0])-1]))

for k in range(len(J)):
   if str(J[k][1])==str("ref.jpg"):
       ref=J[k]

#print(J)


# Comparaison avec l'image de référence

#print(DIFFABS)


print("Temps d'execution :", int(10*(time.time() - start))/10, "secondes")

