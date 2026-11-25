from csv import reader, writer

from matplotlib import image
import rtde_receive
from rtde_receive import RTDEReceiveInterface
import rtde_control 

import ast
import cv2 as cv
import numpy as np 
import socket
import asyncio
import time

# Laatste bericht robot bijhouden, moet globaal vanwege Async functionaliteit 
laatste_robot_bericht = ""

# Posities uitlezen vanuit bestaand document
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

# Bijhouden voor positie en opslaan foto's 
Teller_foto = 0 

async def handle_robot_connection(reader, writer):
    global laatste_robot_bericht

    # Lees data van de robot (max 1024 bytes)
    data = await reader.read(1024)
    bericht = data.decode("utf-8").strip()
    laatste_robot_bericht = bericht

    if bericht == "VRAAG_OM_BEVESTIGING":
        # Handmatige bevestiging voor volgende positie 
        input("Druk op [ENTER] in je terminal om de robot groen licht te geven...") 

        # Terugkoppel naar de robot encoden, (Gemini). 
        antwoord = "GA_VERDER\n"
        writer.write(antwoord.encode("utf-8"))
        await writer.drain()
        print("[SERVER] Bevestiging gestuurd naar de robot!")

    # Kleine pauze 
    writer.close()
    await writer.wait_closed()

async def main():
    # Informatie van de server vaststellen 
    HOST = "192.168.106.199"
    PORT = 5000

    # IP-adres van de robotarm, ingesteld op gegeven controller 
    IP_ADRES = "192.168.106.159"

    # Asynchrone server starten 
    server = await asyncio.start_server(handle_robot_connection, HOST, PORT)
    print(f"Asynchrone server gestart op {HOST}:{PORT}")

    # Eerste verbinding maken met de robotarm
    rtde_c = rtde_control.RTDEControlInterface(IP_ADRES)
    rtde_r = rtde_receive.RTDEReceiveInterface(IP_ADRES)

    # Aspect ratio en resolutie van windows bepalen
    frame_width = 1280
    frame_height = 720
    frame_aspect = frame_width / frame_height

    # Window aanmaken op correcte maat, Wayland dingetje 
    cv.namedWindow('Lookin', cv.WINDOW_NORMAL)
    cv.resizeWindow('Lookin', int(frame_width), int(frame_height))

    # Feedback naar terminal printen 
    print(f"Server en Videoconnectie gestart! Wacht op verbinding van de robot...")

    # RTSP stream van de camera inlezen
    url = "rtsp://admin:admin123@192.168.106.149:554/line=2/stream=5" 
    cam = cv.VideoCapture(url)

    # Main loop
    try:
        while cam.isOpened():
            # Camera uitlezen begin van elke loop 
            ret, frame = cam.read()

            #Cancellen als de camera niet werkt
            if frame is not None: 
                cv.imshow('Lookin', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Gestopt via window")
                break
            
            # Actuele positie van de robot uitlezen via RTDE
            actuele_positie = rtde_r.getActualTCPPose()

            await asyncio.sleep(0.01)  # Kleine pauze om CPU te ontlasten

    finally:
        cam.release()
        cv.destroyAllWindows()
        server.close()
        await server.wait_closed()
        print("Programma succesvol afgesloten.")

# Main loop laten lopen
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript onderbroken door gebruiker.")