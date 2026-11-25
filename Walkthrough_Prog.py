import cv2 as cv 
import numpy as np 
import glob 
import matplotlib.pyplot as plt 
import re

#Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
dist = correction_data["dist_coeffs"]
newcameramtx = correction_data["newcameramtx"]

#Foto's inladen
images_cor = glob.glob('saved_cor/*.png')
images_uncor = glob.glob('saved_uncor/*.png')

#Aspect ratio en resolutie van windows bepalen
frame_width = 1280
frame_height = 720
frame_aspect = frame_width / frame_height

#Het middenpunt van de windows, voor corrected en uncorrected frames bepalen
middenpunt_uncor = np.array([[[frame_width/2, frame_height/2]]], dtype=np.float32)
x_cen_uncor = int(middenpunt_uncor[0][0][0])
y_cen_uncor = int(middenpunt_uncor[0][0][1])

middenpunt_cor = cv.undistortPoints(middenpunt_uncor, mtx, dist, None, newcameramtx)
x_cen_cor = int(middenpunt_cor[0][0][0]) 
y_cen_cor = int(middenpunt_cor[0][0][1])
#print("Middenpunt uncorrected: ", middenpunt_uncor, "Middenpunt corrected: ", middenpunt_cor)

#Uitlijnen van linkerframe 
linkerpunt_frame = np.array([[[0, 0]]], dtype=np.float32)
linkerpunt_frame_cor = cv.undistortPoints(linkerpunt_frame, mtx, dist, None, newcameramtx)
x_lin_cor = int(linkerpunt_frame_cor[0][0][0]) 
y_lin_cor = int(linkerpunt_frame_cor[0][0][1]) 

#Windows aanmaken op correcte maat
cv.namedWindow('Lookin', cv.WINDOW_NORMAL)
cv.resizeWindow('Lookin', int(frame_width), int(frame_height))

#Tracken voor foutcorrectie in algoritme
arealist = []
deviationlist = []
scorelist = []
perimeterlist = []
elipsearealist = []
arearatiolist = []

hoogte_object_list = np.empty(0, dtype = int) 

Zalist = np.empty(0, dtype = int) 
Zblist = np.empty(0, dtype = int) 

afstandlist = np.empty(0, dtype = int) 
labels = []

#Tellen hoeveel bestanden getest 
filecounter = 0
afstandcounter = 0

#Label maken uit filename -> Tutorial van GeeksforGeeks
pattern = '[\w-]+?(?=\.)'

#Upper en lower limits voor mask aanmaken
upper = np.array([255, 255, 255])
lower = np.array([0, 0, 195])

#Kernel aanmaken voor edge detection 
kernel_erode = np.ones((2, 2), np.uint8)
kernel_dilate = np.ones((2, 2), np.uint8)

#Variabelen aanmaken voor het bijhouden van middenpunten van elipsen
x_center_elips_rechterframe = 0 
y_center_elips_rechterframe = 0 

x_center_elips_linkerframe = 0 
y_center_elips_linkerframe = 0

#Variabele voor afstandberekening 
delta_centerpunten = 0

#Voor de berekening van afstand vocale lengte camera uit bestand halen 
fx = mtx[0, 0] 
fy = mtx[0, 1]

#Lenzafstand - gemeten in mm
lenzafstand = 100

basisafstand = 100

#Afstand van camera tot grond in mm
afstand_camera_grond = 500

#Label aanmaken aan de hand van filenaam
def get_label(filename):
    match = re.search(r'/([^/]+?)(\d+)?\.png$', filename)
    
    if match:
        return match.group(1)
    
    return None

#Alle berekeningen uitgevoerd zijn uitgestippeld in schrift
def get_distance(xa, xb): 
    cx = 950.47 / 2
    xa = np.abs(xa - cx)
    xb = np.abs(xb - cx)
        
    #Afstand tot geprojecteerd object 
    AAf = np.sqrt(xa ** 2 + fx ** 2) 
    BBf = np.sqrt(xb ** 2 + fx ** 2)

    #Hoekberekeningen voor beide cameras
    Fi2 = np.arcsin(xb / BBf) 
    Fi1 = np.pi * 0.5 - Fi2

    Theta2 = np.arcsin(xa / AAf)
    Theta1 = np.pi * 0.5 - Theta2

    Alfa = Fi2 + Theta2 

    #Afstand van individuele lenzen tot object
    AP = np.sin(Fi1) * (basisafstand / np.sin(Alfa))
    BP = np.sin(Theta1) * (basisafstand / np.sin(Alfa))

    #Afstand camera tot object
    Za = np.sin(Theta1) * AP
    Zb = np.sin(Fi1) * BP 

    return Za, Zb

#Automatisch door alle fotos heenladen, hier corrected of uncorrected kiezen 
for fname in images_cor: 

    print(fname)
    img = cv.imread(fname) 
    img_hcv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
    H, S, V = cv.split(img_hcv)

    mask = cv.inRange(img_hcv, lower, upper)
    fin = cv.bitwise_and(img, img, mask=mask)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 205, 255, 0)

    eroded = cv.erode(thresh, kernel_erode, iterations=1)
    edged = cv.dilate(eroded, kernel_dilate, iterations=3)

    sobelx = cv.Sobel(edged, cv.CV_64F, 1, 0, ksize=1)  # Horizontal edges
    sobely = cv.Sobel(edged, cv.CV_64F, 0, 1, ksize=1)  # Vertical edges

    # Compute gradient magnitude
    gradient_magnitude = cv.magnitude(sobelx, sobely)
 
    # Convert to uint8
    gradient_magnitude = cv.convertScaleAbs(gradient_magnitude)
    
    contours, _ = cv.findContours(gradient_magnitude, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #Lijn tekenen voor het herkennen van het midden van het window
    cv.line(img, (x_cen_cor, 0), (x_cen_cor, int(frame_height)), (255, 0, 0), 2)

    #Lijn tekenen voor het herkennen van linkerkant van het frame in de window
    cv.line(img, (x_lin_cor, 0), (x_lin_cor, int(frame_height)), (255, 0, 0), 2)
    
    for i, cnt in enumerate(contours):

        if (len(cnt) > 50): 

            ellipse = cv.fitEllipse(cnt)

            (center, axes, angle) = ellipse
            (width, height) = axes
            (x_cen, y_cen) = center

            ratio = min(width, height) / max(width, height)
            deviation = 1 - ratio

            area = cv.contourArea(cnt)
            perimeter = cv.arcLength(cnt, False)

            ellipse_area = np.pi * (width / 2) * (height / 2)

            score = 4 * np.pi * ellipse_area / (perimeter ** 2)
            
            #Poging tot uitfilteren van te grote contouren 
            #Aan de hand van grootte kleur toekennen 
            if (area < 200) or (19000 < area < 19000) or (deviation > 0.6) or (deviation < 0.2) or (score < 0.19):
                cv.ellipse(img, ellipse, (0,0,255), 1) #Afgekeurde contouren

                arealist.append(cv.contourArea(cnt))
                deviationlist.append(deviation)
                scorelist.append(score)
                perimeterlist.append(perimeter)
                elipsearealist.append(ellipse_area)

            else: 
                #Uitvogelen of het in het linker of rechterplaatje zit
                if (x_cen > x_cen_cor): 
                    x_center_elips_rechterframe = x_cen - x_cen_cor
                    y_center_elips_rechterframe = y_cen - y_cen_cor

                    cv.circle(img, (int(x_center_elips_rechterframe), int(y_cen)), 2, (0, 255, 255), 3)
                else: 
                    x_center_elips_linkerframe = x_cen - x_lin_cor 
                    y_center_elips_linkerframe = y_cen - y_lin_cor

                    cv.circle(img, (int(x_center_elips_linkerframe), int(y_cen)), 2, (0, 255, 255), 3)

                cv.ellipse(img, ellipse, (0, 255, 0), 3)
                cv.circle(img, (int(x_cen), int(y_cen)), 2, (255, 0, 0), 3)

                # print("coordinaten rechts: ", x_center_elips_rechterframe, " - ", y_cen, "\n")
                # print("coordinaten links: ", x_center_elips_linkerframe, " - ", y_cen, "\n")

    if ((x_center_elips_rechterframe) and (x_center_elips_linkerframe)): 

        print("coordinaten rechts: ", x_center_elips_rechterframe, " - ", y_cen, "\n")
        print("coordinaten links: ", x_center_elips_linkerframe, " - ", y_cen, "\n")

        delta_centerpunten = np.abs(x_center_elips_linkerframe - x_center_elips_rechterframe)
        afstand_tot_object = (fx * lenzafstand) / delta_centerpunten

        hoogte_object_list = np.append(hoogte_object_list, int(afstand_camera_grond - afstand_tot_object))
        
        afstandlist = np.append(afstandlist, int(afstand_tot_object))

        lengteZa, lengteZb = get_distance(x_center_elips_linkerframe, x_center_elips_rechterframe)

        Zalist = np.append(Zalist, int(lengteZa))
        Zblist = np.append(Zblist, int(lengteZb))

        afstandcounter += 1
        labels.append(get_label(fname))

        print("Delta tussen coordinaten (Pixels): ", np.abs(x_center_elips_linkerframe - x_center_elips_rechterframe))
        print("Afstand tot object (mm): ", afstand_tot_object)
        print("Hoogte gedetecteerd object (mm): ", (afstand_camera_grond - afstand_tot_object))

        print("\nAfstand tot object (mm): ", lengteZa)
        print("Afstand tot object (mm): ", lengteZb)

    # cv.imshow('Lookin', img)
    # cv.imshow('Sobel', gradient_magnitude)
    # cv.imshow('Edged', edged)

    # cv.imshow('H', H)
    # cv.imshow('S', S)
    # cv.imshow('V', V)

    print("\n", fname, " has ", len(contours), " contours,", "Contour surfaces: ", arealist, 
          "\nDeviation is as follows: ", deviationlist, 
          "\nScore is: ", scorelist, 
          "\nPerimeters are as follows: ", perimeterlist, 
          "\nArea is: ", elipsearealist, 
          "\n\n")

    arealist.clear()
    deviationlist.clear()
    scorelist.clear()
    perimeterlist.clear()
    elipsearealist.clear()
    arearatiolist.clear()

    x_center_elips_rechterframe = 0
    x_center_elips_linkerframe = 0

    filecounter += 1
    #cv.waitKey(0)

print("Hoeveelheid fotos getest: ", filecounter, 
      "\nHoeveelheid fotos met afstand: ", afstandcounter, 
      "\nAlle filenamen:", labels, 
      "\n\nAlle objecthoogtes:", hoogte_object_list,
      "\n\nAlle afstanden (Nieuw):", Zalist, 
      "\n\nAlle afstanden (Oud):", afstandlist) 
      
for i in range(len(labels)): 
    plt.figure(1)
    plt.scatter(labels[i], Zalist[i])
    plt.xlabel("Label")
    plt.ylabel("Afstand")
    plt.title("Nieuwe")
    plt.axis([-1, 9, 200, 550])

for i in range(len(labels)): 
    plt.figure(2)
    plt.scatter(labels[i], afstandlist[i])
    plt.xlabel("Label")
    plt.ylabel("Afstand")
    plt.title("Oude")
    plt.axis([-1, 9, 200, 550])

for i in range(len(labels)): 
    plt.figure(3)
    plt.scatter(Zalist[i], afstandlist[i])
    plt.xlabel("Nieuw")
    plt.ylabel("Oud")
    plt.title("Afstand VS")
    plt.axis([200, 550, 200, 550])
plt.show()

#Wachten tot een toetsdruk voor het eindigen van het loop
cv.waitKey(0)

                #Checkvragen nakijken
                # if ((cv.contourArea(cnt) < 14000) or (cv.contourArea(cnt) > 19000)) and ((0.2 < deviation < 0.5)) and (score < 0.25):
                #     print("Bad checking? : ", fname, "Area: ", cv.contourArea(cnt), "Deviation: ", deviation, "Score: ", score)
                #     print("Perimeter:", perimeter, "Expected area: ", perimeter ** 2 / (4 * np.pi), " W/H: ", width, " / ", height)