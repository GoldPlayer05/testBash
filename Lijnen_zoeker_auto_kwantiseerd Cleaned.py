import cv2 as cv 
import numpy as np 
import glob 
import matplotlib.pyplot as plt
import re
import math
from collections import defaultdict


# v1 lijnen/oren zoeken haalt nu ongeveer 1/3 van de steelpannen eruit en 1 kookpan(slecht)
# kijk of dit beter kan en kijk waarom de andere afgekeurd zijn.
# mediaan geprobeert dat vernageld alles waardoor je meer error krijgt 
# strenger naar de hoeken gekeken dat werkt semi aangezien hij een deel van de goede hoeken afkeurd door de ene uitwijker. 
# mogelijk een ding voor pre-processing 
# lengte is leuk maar zegt uiteindelijk weinig. 
# uiteindelijk is er gekozen voor de o.1 filter omdat dat minder fout gaf met bijna dezelfde fout ong 91% accuratie


#Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
dist = correction_data["dist_coeffs"]
newcameramtx = correction_data["newcameramtx"]

#Foto's inladen
images_cor = glob.glob('./saved_cor/*.png')
images_uncor = glob.glob('./saved_uncor/*.png')

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
# cv.namedWindow('Lookin', cv.WINDOW_NORMAL)
# cv.resizeWindow('Lookin', int(frame_width), int(frame_height))

hoogte_object_list = np.empty(0, dtype = int) 

Zalist = np.empty(0, dtype = int) 
Zblist = np.empty(0, dtype = int) 

afstandlist = np.empty(0, dtype = int) 
labels = []

#Label maken uit filename -> Tutorial van GeeksforGeeks
pattern = '[\w-]+?(?=\.)'

#Upper en lower limits voor mask aanmaken
upper = np.array([255, 255, 255])
lower = np.array([0, 0, 195])

#Kernel aanmaken voor edge detection 
kernel_erode = np.ones((2, 2), np.uint8)
kernel_dilate = np.ones((2, 2), np.uint8)


#Voor de berekening van afstand vocale lengte camera uit bestand halen 
fx = mtx[0, 0] 
fy = mtx[0, 1]



avg_lengte_array = []
Hoek_lengte_array = []
Hoek_acceptatie_array = []
def get_label(filename):
    match = re.search(r'/([^/]+?)(\d+)?\.png$', filename)
    
    if match:
        return match.group(1)
    
    return None


def Find_lines(images_cor):
    img = cv.imread(images_cor)

    img_hcv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(img_hcv)

    mask = cv.inRange(img_hcv, lower, upper)
    fin = cv.bitwise_and(img, img, mask=mask)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 205, 255, 0)

    eroded = cv.erode(thresh, kernel_erode, iterations=1)
    edged = cv.dilate(eroded, kernel_dilate, iterations=3)

    sobelx = cv.Sobel(edged, cv.CV_64F, 1, 0, ksize=1)
    sobely = cv.Sobel(edged, cv.CV_64F, 0, 1, ksize=1)

    gradient_magnitude = cv.magnitude(sobelx, sobely)
    gradient_magnitude = cv.convertScaleAbs(gradient_magnitude)

    # ---------------- HOUGH LINES ----------------
    display_img = img.copy()
    avg_lengte = []
    Hoek_array = []
    l = []
    
    #Hough lines Propabilistic 
    Label = get_label(Img)
    linesP = cv.HoughLinesP(gradient_magnitude, 1, np.pi / 180, 83, None, 50, 15)

    if linesP is not None:
        for l in linesP:
            l = l.flatten()
            x1, y1, x2, y2 = map(int, l)
            cv.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)
            #als hij minder dan 2 lines ziet niet meenemen dan ziet hij strays
            if(len(linesP) > 2):
                lengte = math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
                hoek = math.acos(abs(x1-x2)/lengte)
                avg_lengte.append(lengte)
                Hoek_array.append(hoek)
                Hoek_lengte_array.append((Label, lengte, hoek))

        #kijk of de lengte van de lengte-/hoek_array 0 is anders avg pakken.
        if((len(avg_lengte) != 0) or (len(Hoek_array) != 0)):   #hier keurt hij 1 lijn af.
            avg_lengte_array.append((Label, (sum(avg_lengte)/len(avg_lengte))))    
            
            Hoek_overeenkomend = []
            gem_hoek = (sum(Hoek_array)/len(Hoek_array))
            
            for Index_hoek in range(len(Hoek_array)):
                # radialen YIPPIE :(
                abs_hoek = abs((gem_hoek)-Hoek_array[Index_hoek])
                if(abs_hoek < 0.07 ):#deze aanpassen als je strenger wilt filteren of een hoek in de buurt zit van de anderen.
                    Hoek_overeenkomend.append(Hoek_array[Index_hoek])
                    
            if(len(Hoek_overeenkomend) >= 2):
                Hoek_acceptatie_array.append((Label,gem_hoek))
                return (Label,gem_hoek) #hier zetten wat hij moet krijgen om te zeggen ja ik heb een steelpan/koekepan
    return -1
   
for Img in images_cor:
    # (label, gem_hoek)
    Result = Find_lines(Img)
    print( Result)

Steelpan_handle=0
koekenpan_handle=0
overig_handle=0
for i in range(len(Hoek_acceptatie_array)): 
            L, x = Hoek_acceptatie_array[i]
            plt.figure(3)
            plt.scatter(L, x)
            plt.xlabel("label")
            plt.ylabel("Hoek")
            plt.title("Hoek acceptatie ")
            plt.axis([-1, 4, 0, 2.5])
            if(L == 'saved_cor\cor_steelpan'):
                Steelpan_handle =  Steelpan_handle+1
            elif(L == 'saved_cor\cor_koekenpan'):
                koekenpan_handle=koekenpan_handle+1
            else:
                overig_handle=overig_handle+1
print("steelpan",Steelpan_handle)
print("koekenpan",koekenpan_handle)
print("overig/fout",overig_handle)
                    
plt.show()

cv.waitKey(0)
