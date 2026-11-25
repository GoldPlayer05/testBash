import rtde_receive
from rtde_receive import RTDEReceiveInterface
import rtde_control 

import ast
import cv2 as cv
import numpy as np 
import glob 
import math 
import time 

# IP-adres van de robotarm, ingesteld op gegeven controller 
IP_ADRES = "192.168.106.159"

# Eerste verbinding maken met de robotarm
# rtde_c = rtde_control.RTDEControlInterface(IP_ADRES)
rtde_r = rtde_receive.RTDEReceiveInterface(IP_ADRES)

# Bijhouden welke positie is bereikt
teller = 0

print("Guten tag!") 

url = "rtsp://admin:admin123@192.168.106.149:554/line=2/stream=5" 
cam = cv.VideoCapture(url) 

#Aspect ratio en resolutie van windows bepalen
frame_width = 1280
frame_height = 720
frame_aspect = frame_width / frame_height

#Windows aanmaken op correcte maat
cv.namedWindow('Lookin', cv.WINDOW_NORMAL)
cv.resizeWindow('Lookin', int(frame_width), int(frame_height))

# Gevrichten map 
"""
Gevricht 1: Schouder x,y vlak
Gevricht 2: Schouder z,y vlak 
Gevricht 3: Elleboog 
Gevricht 4: Pols tilt 
Gevricht 5: Pols roll 
Gevricht 6: Hoofd (laatste)
"""

# Posities noteren
alle_posities = []

with open("test_resultaten_PERM.txt", "r") as file:
    for line in file:
        if "[" in line and "]" in line:

            # Begin en eindindexen localiseren
            start_index = line.find("[")
            eind_index = line.rfind("]") + 1
            
            lijst_als_tekst = line[start_index:eind_index]
            getallen_lijst = ast.literal_eval(lijst_als_tekst)

            alle_posities.append(getallen_lijst)
    print("Het zijn er: ", len(alle_posities))

print("Bewegen naar startpositie")

try: 
    while True: 
        ret, frame = cam.read() 

        if not ret: 
            print("Camera doet 't niet!")
            break 
        
        cv.imshow('Lookin', frame)

        # actuele_positie = rtde_r.getActualTCPPose()

        # if (actuele_positie == alle_posities[teller]): 
        #     print(f"Positie {teller} bereikt!") 
        #     teller += 1 
        #     time.sleep(1) 

        cv.waitKey(1)

except KeyboardInterrupt: 
    print("Programma gestopt door gebruiker.") 