import rtde_receive
from rtde_receive import RTDEReceiveInterface
import rtde_control 

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

# Bijhouden welke positie wordt opgeslagen
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

try: 
    while True: 
        ret, frame = cam.read() 

        if not ret: 
            print("Camera doet 't niet!")
            break 

        cv.imshow('Lookin', frame)

        actuele_positie = rtde_r.getActualTCPPose()

        # print("\n\nActuele TCP-positie:", actuele_positie[0], actuele_positie[1], actuele_positie[2], "\n")
        # print("Actuele TCP-oriëntatie:", actuele_positie[3], actuele_positie[4], actuele_positie[5], "\n")
        
        # joints_rad = rtde_r.getActualQ() 
        # print("Gevrichten hoeken (radialen):", joints_rad, "\n")

        # for i in range(len(joints_rad)):
        #     joints_deg = [math.degrees(angle) for angle in joints_rad]
        #     print(f"Gevricht {i+1} hoek (graden): {joints_deg[i]}")

        # time.sleep(1)  

        if cv.waitKey(1) == ord('s'):
            teller += 1
            with open("test_resultaten.txt", "a") as file:
                file.write(f"Positie en oriëntatie van de TCP in positie {teller}: {actuele_positie}\n")
            print(f"Positie en oriëntatie van de TCP in positie {teller} opgeslagen in test_resultaten.txt")

except KeyboardInterrupt:
    print("Programma gestopt door gebruiker.")