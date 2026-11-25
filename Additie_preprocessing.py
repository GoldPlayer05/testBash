import rtde_receive
from rtde_receive import RTDEReceiveInterface
import rtde_control 

import numpy as np
import ast
import cv2 as cv
import socket
import threading 
import time

# Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
dist = correction_data["dist_coeffs"]
newcameramtx = correction_data["newcameramtx"]

# Laatste bericht robot bijhouden, moet globaal vanwege Async functionaliteit 
laatste_robot_bericht = ""

# Data veilig delen tussen threads 
data_lock = threading.Lock()

# Variabele om te controleren of er een foto moet worden genomen
foto_trigger = False 

# Gaan voor volgende positie 
go = False 

# Bijhouden voor positie en opslaan foto's 
Teller_foto = 0 

# Posities uitlezen vanuit bestaand document, wordt niet benut 
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

# Functie voor afhandelen connectie met de robotarm 
# Deze is oneinding, en gebeurt op een aparte thread
def handle_robot_connection(HOST, PORT):
    global laatste_robot_bericht, Teller_foto, foto_trigger, go

    # Server afstellen voor verbinden met arm 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    # Feedback terminal 
    print(f"\n\nServer gestart! Wacht op verbinding van de robot...")

    # Main loop van de functie 
    try: 
        while True:
            conn, addr = server_socket.accept()
            print(f"\nVerbonden met robot! Wachten op bericht...")
    
            # Data decoden van de robot 
            data = conn.recv(1024).decode("utf-8").strip()

            # Data lock voor gedeelde variabelen 
            with data_lock: 
                laatste_robot_bericht = data

            # Feedback terminal
            print(f"[Aparte Thread] Ontvangen van robot: '{data}'")

            # Begin fotoprocess 
            if data == "VRAAG_OM_BEVESTIGING":
                # Handmatig de volgende positie goedkeuren 
                input("\nDruk op [ENTER] in je terminal om de robot groen licht te geven")

                # Starten wachten tot foto
                # while True: 
                #     with data_lock: 
                #         if go: 
                #             break 
                #     time.sleep(0.05)

                # with data_lock: 
                #     go = False

                # Handshake terug naar de robot encoden, (Gemini).
                antwoord = "GA_VERDER\n"
                conn.sendall(antwoord.encode("utf-8"))
                print("Bevestiging gestuurd naar de robot!\n")

            elif data == "VOLGEND_WAYPOINT_BEREIKT": 
                foto_trigger = True 
                Teller_foto += 1
                print(f"Robot heeft waypoint {Teller_foto} bereikt!") 

            conn.close()
    except Exception as e:
        print(f"Fout in robotverbinding: {e}")
    finally:
        server_socket.close() 

# Alle berekeningen die uitgevoerd zijn, zijn uitgestippeld in schrift
# Dit is een oproepbare functie 
def get_distance(xa, xb): 
    # Aantal variabelen voor afstandberekening 
    basisafstand = 100

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

# Functie voor het voorbereiden van frames voor herkenning 
# Dit is een oproepbare functie 
def preprocessing(frame): 
    # Variabelen voor pre-processing 
    #------------------------------------------------------------------------------------------------------
     # Upper en lower limits voor mask aanmaken
    upper = np.array([255, 255, 255])
    lower = np.array([0, 0, 195])

    # Kernels aanmaken voor edge detection 
    kernel_erode = np.ones((2, 2), np.uint8)
    kernel_dilate = np.ones((2, 2), np.uint8)
    #------------------------------------------------------------------------------------------------------

    # Begin pre-processing 
    #------------------------------------------------------------------------------------------------------
    img = frame
    img = cv.undistort(frame, mtx, dist, None, newcameramtx)
    img_hcv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  

    mask = cv.inRange(img_hcv, lower, upper)
    fin = cv.bitwise_and(img, img, mask=mask)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 155, 255, 0)

    dilate = cv.dilate(thresh, kernel_dilate, iterations=65)
    eroded_voor_diepte = cv.erode(dilate, kernel_erode, iterations=65)

    dilated_voor_solidity = cv.dilate(thresh, kernel_dilate, iterations=3)

    # Twee keer sobel edge, een keer voor de diepte, een keer voor de solidity 
    sobelx_diepte = cv.Sobel(eroded_voor_diepte, cv.CV_64F, 1, 0, ksize=1)  # Horizontal edges
    sobely_diepte = cv.Sobel(eroded_voor_diepte, cv.CV_64F, 0, 1, ksize=1)  # Vertical edges

    sobelx_solidity = cv.Sobel(dilated_voor_solidity, cv.CV_64F, 1, 0, ksize=1)  # Horizontal edges
    sobely_solidity = cv.Sobel(dilated_voor_solidity, cv.CV_64F, 0, 1, ksize=1)  # Vertical edges

    # Compute gradient magnitude
    gradient_diepte = cv.magnitude(sobelx_diepte, sobely_diepte)
    gradient_solidity = cv.magnitude(sobelx_solidity, sobely_solidity)
    
    # Convert to uint8
    gradient_diepte = cv.convertScaleAbs(gradient_diepte)
    gradient_solidity = cv.convertScaleAbs(gradient_solidity)
    #------------------------------------------------------------------------------------------------------

    # Teruggeven van behandelde frames 
    #------------------------------------------------------------------------------------------------------
    return gradient_diepte, gradient_solidity
    #------------------------------------------------------------------------------------------------------

# Beginnen main functieloop     
def main():
    global laatste_robot_bericht, Teller_foto, foto_trigger, go

    # Timen van de foto nemen 
    starttijd = None 
    wachttijd = 2.5
    opslaan = None

    # Informatie van de server vaststellen 
    HOST = "192.168.106.199"
    PORT = 5000

    # IP-adres van de robotarm, ingesteld op gegeven controller 
    IP_ADRES = "192.168.106.159"

    # Asynchrone server starten 
    socket_thread = threading.Thread(target=handle_robot_connection, args=(HOST, PORT), daemon=True)
    socket_thread.start()
    print(f"Andere Thread server gestart op {HOST}:{PORT}")

    # Eerste verbinding maken met de robotarm
    # rtde_c = rtde_control.RTDEControlInterface(IP_ADRES)
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
        while True:
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

            # Met data lock foto trigger opslaan in locale variabele
            with data_lock: 
                if foto_trigger: 
                    opslaan = True 
                    foto_trigger = False  # Reset de trigger na het opslaan

            # Starttijd eenmalig (per cyclus) opslaan 
            if opslaan and starttijd is None:
                starttijd = time.time()  

            # Timen voor foto vastleggen, vanwege delay cameraverbinding 
            if starttijd is not None: 
                verlopen_tijd = time.time() - starttijd

            # Foto maken, timer resetten en dataopslag legen 
            if opslaan and wachttijd < verlopen_tijd: 
                cv.imwrite(f"./testfotos/foto_positie_{Teller_foto}.png", frame)
                print(f"\nFoto gemaakt({Teller_foto})!\n")

                processed_frame, _ = preprocessing(frame)
                cv.imshow('Klaar?', processed_frame)
                
                starttijd = None
                opslaan = False

                with data_lock: 
                    go = True

            # Ongebruike optie 
            with data_lock:
                if laatste_robot_bericht:
                    pass 

    # Gepast afsluiten programma 
    finally:
        cam.release()
        cv.destroyAllWindows()
        print("Programma afgesloten.")

# Main loop laten lopen
if __name__ == "__main__":
    main()


""" Later invoegen 
# Begin sectie declaraties voor rechtrekken posities op het frame 
# ------------------------------------------------------------------------------------------------------
# Afmetingen van frame bijleveren 
frame_width = 1280
frame_height = 720
frame_aspect = frame_width / frame_height

# Het middenpunt van de windows, voor corrected en uncorrected frames bepalen
middenpunt_uncor = np.array([[[frame_width/2, frame_height/2]]], dtype=np.float32)
x_cen_uncor = int(middenpunt_uncor[0][0][0])
y_cen_uncor = int(middenpunt_uncor[0][0][1])

middenpunt_cor = cv.undistortPoints(middenpunt_uncor, mtx, dist, None, newcameramtx)
x_cen_cor = int(middenpunt_cor[0][0][0]) 
y_cen_cor = int(middenpunt_cor[0][0][1])

# Uitlijnen van linkerframe 
linkerpunt_frame = np.array([[[0, 0]]], dtype=np.float32)
linkerpunt_frame_cor = cv.undistortPoints(linkerpunt_frame, mtx, dist, None, newcameramtx)
x_lin_cor = int(linkerpunt_frame_cor[0][0][0]) 
y_lin_cor = int(linkerpunt_frame_cor[0][0][1]) 
#------------------------------------------------------------------------------------------------------

# Begin sectie declaraties voor afstanddetectie dmv witte pixels 
#------------------------------------------------------------------------------------------------------
# Witte pixels bijhouden 
x_wit_rechts = np.empty(0, dtype = int) 
x_wit_links = np.empty(0, dtype = int) 

x_test_wit_r = np.empty(0, dtype = int) 
x_test_wit_l = np.empty(0, dtype = int) 

Zalist_wit = np.empty(0, dtype = int) 
Zblist_wit = np.empty(0, dtype = int) 

Zalist_wit_totaal = np.empty(0, dtype = int) 
Zblist_wit_totaal = np.empty(0, dtype = int) 
#------------------------------------------------------------------------------------------------------

# Aantal variabelen voor bijhouden van centerpunten 
#------------------------------------------------------------------------------------------------------
# Variabelen aanmaken voor het bijhouden van middenpunten van elipsen
x_center_elips_rechterframe = 0 
y_center_elips_rechterframe = 0 

x_center_elips_linkerframe = 0 
y_center_elips_linkerframe = 0

# Variabele voor afstandberekening 
delta_centerpunten = 0

# Voor de berekening van afstand vocale lengte camera uit bestand halen 
fx = mtx[0, 0] 
fy = mtx[0, 1]

# Afstand van camera tot grond in mm
afstand_camera_grond = 500
#------------------------------------------------------------------------------------------------------


"""