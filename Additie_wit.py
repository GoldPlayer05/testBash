import rtde_receive
from rtde_receive import RTDEReceiveInterface
import rtde_control 

from matplotlib import pyplot as plt
import numpy as np
import ast
import cv2 as cv
import socket
import threading 
import time

# Dit is een aanpassing voor alleen de test branch

# Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
dist = correction_data["dist_coeffs"]
newcameramtx = correction_data["newcameramtx"]

# Begin sectie declaraties voor het sorteren van punten tussen linker- en rechterplaatje 
# Deze is global vanwege gebruik in diverse functies 
# ------------------------------------------------------------------------------------------------------
# Afmetingen van frame bijleveren 
frame_width = 1280
frame_height = 720

# Het middenpunt van het frame bepalen, eerst normaal, dan gecorrigeerd 
middenpunt_uncor = np.array([[[frame_width/2, frame_height/2]]], dtype=np.float32)
middenpunt_cor = cv.undistortPoints(middenpunt_uncor, mtx, dist, None, newcameramtx)
x_cen_cor = int(middenpunt_cor[0][0][0]) 

# Uitlijnen van linkerframe, deze is ook anders dan bij de ongecorrigeerde foto 
linkerpunt_frame = np.array([[[0, 0]]], dtype=np.float32)
linkerpunt_frame_cor = cv.undistortPoints(linkerpunt_frame, mtx, dist, None, newcameramtx)
x_lin_cor = int(linkerpunt_frame_cor[0][0][0]) 

# Voor de berekening van afstand vocale lengte camera uit bestand halen 
fx = mtx[0, 0] 
fy = mtx[0, 1]
# ------------------------------------------------------------------------------------------------------

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
    print(f"\n\nServer gestart! Wachten op verbinding met de robot")

    # Main loop van de functie 
    try: 
        while True:
            conn, addr = server_socket.accept()
            print(f"\nVerbonden met robot! Wachten")
    
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
                input("\nDruk op [ENTER] voor volgende positie")

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
    global mtx, dist, newcameramtx
    
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
    img = cv.undistort(img, mtx, dist, None, newcameramtx)
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
    return gradient_diepte, gradient_solidity, eroded_voor_diepte
    #------------------------------------------------------------------------------------------------------

# Functie voor het bepalen van de afstand dmv cv.distancetransform
def euclidian_afstand(eroded_voor_diepte): 
    global mtx, dist, newcameramtx, fx, x_cen_cor

    #----------------------------------------------------------------------------
    # Euclidian afstand van pixels tot NUL-pixels bepalen
    distance = cv.distanceTransform(eroded_voor_diepte, cv.DIST_L2, 5)

    dist_links = distance[:, :x_cen_cor]
    dist_rechts = distance[:, x_cen_cor:]

    # Gemaakt plaatje tekenbaar door opencv maken 
    troubleshoot_euc = cv.normalize(dist, None, 0, 1.0, cv.NORM_MINMAX)

    # Zoek de pixel met de grootste afstand
    _, max_radius_links, _, center_links = cv.minMaxLoc(dist_links)
    _, max_radius_rechts, _, center_rechts = cv.minMaxLoc(dist_rechts)

    # Middelpunt en straal
    x_cen_links, y_cen_links = center_links
    x_cen_rechts, y_cen_rechts = center_rechts

    radius_links = max_radius_links
    radius_rechts = max_radius_rechts

    # print("Delta test: ", int(x_cen_links) - int(x_cen_rechts), "\n")

    afstand_Za, afstand_Zb = get_distance(x_cen_links, x_cen_rechts)
    #----------------------------------------------------------------------------

    # Gewilde variabelen terugleveren 
    #----------------------------------------------------------------------------
    return afstand_Za, afstand_Zb, radius_links, radius_rechts, troubleshoot_euc
    #----------------------------------------------------------------------------

def wittepunten_afstand(gradient_diepte): 
    global x_cen_cor, x_lin_cor

    # Witte pixels bijhouden 
    x_wit_rechts = np.empty(0, dtype = int) 
    x_wit_links = np.empty(0, dtype = int) 

    x_test_wit_r = np.empty(0, dtype = int) 
    x_test_wit_l = np.empty(0, dtype = int) 

    # Coordinaten van witte pixels vinden in de meegegeven foto  
    coordinates = np.argwhere(gradient_diepte == 255)

    # Alle Y-coordinaten vinden 
    y_values = np.unique(coordinates[:, 0])

    # Troubleshootwaarden 
    gem_y = np.mean(coordinates[:, 0])
    gem_dev = np.std(coordinates[:, 1])

    # Alle gevonden punten sorteren tussen de foto's 
    # Dit gebeurt per Y-laag, ik heb inspiratie genomen van de VGA aansturing 
    for i in y_values: 
        x_values = np.where(gradient_diepte[i, :] > 0)[0]
        
        for l in range(len(x_values)): 
            if (x_values[l] > x_cen_cor): 
                x_test_wit_r = np.append(x_test_wit_r, (x_values[l]))
                x_wit_rechts = np.append(x_wit_rechts, (x_values[l] - x_cen_cor))
            else: 
                x_test_wit_l = np.append(x_test_wit_l, (x_values[l]))
                x_wit_links = np.append(x_wit_links, (x_values[l] - x_lin_cor))

    # For loop geeft per gevonden pixelpaar een output, dit moet gemiddeld worden 
    gem_x_l = np.mean(x_wit_links) 
    gem_x_r = np.mean(x_wit_rechts)

    # Maximale en minimale x coordinaten per foto 
    x_l_l = np.min(x_wit_links)
    x_l_r = np.max(x_wit_links)
    x_r_l = np.min(x_wit_rechts)
    x_r_r = np.max(x_wit_rechts)

    # print("\ncoordinaten rechts: ", gem_x_r)
    # print("coordinaten links: ", gem_x_l)
    # print("Delta tussen coordinaten (Pixels): ", int(gem_x_l - gem_x_r))

    # Berekenen van de afstand met de gevonden coordinaten 
    lengteZa_witL, lengteZb_witL = get_distance(int(x_l_l), int(x_r_l))
    lengteZa_witR, lengteZb_witR = get_distance(int(x_l_r), int(x_r_r))

    return lengteZa_witL, lengteZb_witL, lengteZa_witR, lengteZb_witR

def fotosplitter(frame): 
    global x_cen_cor, x_lin_cor 
    # x_lin_cor = 30, delta = 716 
    # x_cen_cor = 746, delta = 534 

    x_lin_spoof = 212
    
    frame_width = 1280
    frame_height = 720

    if frame is None or frame.size == 0: 
        print("Fout in frame")
        return None, None 

    linker_frame = frame[0: frame_height, x_lin_spoof:x_cen_cor]
    rechter_frame = frame[0: frame_height, x_cen_cor:frame_width]

    linker_frame = cv.cvtColor(linker_frame, cv.COLOR_BGR2GRAY)
    rechter_frame =  cv.cvtColor(rechter_frame, cv.COLOR_BGR2GRAY)

    return linker_frame, rechter_frame 

def platformvinder(imgL, imgR): 
    stereo = cv.StereoSGBM.create(minDisparity=0, numDisparities=112, blockSize=15)
    disparity = stereo.compute(imgL,imgR) 

    return disparity

# Beginnen main functieloop     
def main():
    global laatste_robot_bericht, Teller_foto, foto_trigger, go, mtx, dist, newcameramtx

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

    # Variabelen voor het tracken van afstand dmv methode cv.distanceTransform
    Zalist_euc = np.empty(0, dtype = int) 
    Zblist_euc = np.empty(0, dtype = int) 
    Radius_euc = np.empty(0, dtype = int) 

    # Variabelen voor het tracken van afstand dmv methode wittepuntendetectie 
    Zalist_wit = np.empty(0, dtype = int) 
    Zblist_wit = np.empty(0, dtype = int) 

    # Window aanmaken op correcte maat, Wayland dingetje 
    cv.namedWindow('Lookin', cv.WINDOW_NORMAL)
    cv.resizeWindow('Lookin', int(frame_width), int(frame_height))

    # Feedback naar terminal printen 
    print(f"Server en Videoconnectie gestart! Wacht op verbinding van de robot...")

    # RTSP stream van de camera inlezen
    url = "rtsp://admin:admin123@192.168.106.149:554/line=1/stream=1" 
    cam = cv.VideoCapture(url)

    

    # Main loop
    try:
        while True:
            # Camera uitlezen begin van elke loop 
            ret, frame = cam.read()

            frame_cor = cv.undistort(frame, mtx, dist, None, newcameramtx)

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

                # Frame door de pre-processing gooien
                processed_frame, _, pre_euc_doorgeef = preprocessing(frame)
                # cv.imshow('Klaar?', processed_frame)

                # Eerste afstandsmethode toepassen 
                a, b, c1, c2, _ = euclidian_afstand(pre_euc_doorgeef) 

                # Berekende waarden opslaan 
                Zalist_euc = np.append(Zalist_euc, int(a))
                Zblist_euc = np.append(Zblist_euc, int(b))
                Radius_euc = np.append(Radius_euc, int((c1 + c2)/2))

                # Tweede afstandsmethode toepassen 
                l1, l2, r1, r2 = wittepunten_afstand(processed_frame)

                # Gemiddelden van de teruggekregen waarden pakken 
                l = int((l1 + l2) / 2)
                r = int((r1 + r2) / 2)

                # Gemiddeld opslaan voor plotten 
                Zalist_wit = np.append(Zalist_wit, l)
                Zblist_wit = np.append(Zblist_wit, r)

                # Stereo afstand bepalen voor zoning, deze is exclusief aan deze samenvoeging 
                imgL, imgR = fotosplitter(frame_cor)

                cv.imshow('Links', imgL)
                cv.imshow('Rechts?', imgR)

                des = platformvinder(imgL, imgR)

                des_nor = cv.normalize(
                                                    src=des, 
                                                    dst=None, 
                                                    alpha=0, 
                                                    beta=255, 
                                                    norm_type=cv.NORM_MINMAX, 
                                                    dtype=cv.CV_8U)

                plt.imshow(des_nor,'gray')
                plt.show()

                # Variabelen terugzetten voor volgende loop 
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
# Begin sectie declaraties voor afstanddetectie dmv witte pixels 
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
#------------------------------------------------------------------------------------------------------


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