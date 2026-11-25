import rtde_receive
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R
import rtde_control 

from matplotlib import pyplot as plt
import numpy as np
import ast
import cv2 as cv
import socket
import threading 
import time
import math

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

# Frame-beperking toevoegen 

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
fy = mtx[1, 1]
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
def robot_connectie(HOST, PORT):
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
                # input("\nDruk op [ENTER] voor volgende positie")

                # Starten wachten tot foto
                while True: 
                    with data_lock: 
                        if go: 
                            break 
                    time.sleep(0.05)

                with data_lock: 
                    go = False

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
def get_distance(xa, xb, hoogte_camera): 
    # Aantal variabelen voor afstandberekening 
    basisafstand = 100

    # Het principal point, optisch midden, is anders met de bounding boxes. 
    # cx = 950.47 / 2
    cx = 175.47
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

    Sa = hoogte_camera - Za
    Sb = hoogte_camera - Zb

    return Sa, Sb

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
    # img = cv.undistort(img, mtx, dist, None, newcameramtx)

    img_hcv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  

    mask = cv.inRange(img_hcv, lower, upper)
    fin = cv.bitwise_and(img, img, mask=mask)

    # Thresholden voor beide solidity en distancetransform, deze werken het best op verschillende waarden 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 205, 255, 0)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, threshed = cv.threshold(gray, 195, 255, 0)

    cv.imshow('Thresh', thresh)

    eroded_lijn = cv.erode(thresh, kernel_erode, iterations=1)
    edged_lijn = cv.dilate(eroded_lijn, kernel_dilate, iterations=3)

    sobelx_lijn = cv.Sobel(edged_lijn, cv.CV_64F, 1, 0, ksize=1)
    sobely_lijn = cv.Sobel(edged_lijn, cv.CV_64F, 0, 1, ksize=1)

    gradient_magnitude_lijn = cv.magnitude(sobelx_lijn, sobely_lijn)
    gradient_magnitude_lijn = cv.convertScaleAbs(gradient_magnitude_lijn)

    dilate = cv.dilate(thresh, kernel_dilate, iterations=10)
    eroded_voor_diepte = cv.erode(dilate, kernel_erode, iterations=10)

    # cv.imshow('dilate', dilate)
    # cv.imshow('eroded_voor_diepte', eroded_voor_diepte)

    dilated_voor_solidity = cv.dilate(threshed, kernel_dilate, iterations=3)

    # cv.imshow('dilated_voor_solidity', dilated_voor_solidity)

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
    return gradient_diepte, gradient_solidity, eroded_voor_diepte, gradient_magnitude_lijn
    #------------------------------------------------------------------------------------------------------

# Functie voor het bepalen van de afstand dmv cv.distancetransform
def euclidian_afstand(eroded_voor_diepte, hoogte_camera): 
    global mtx, dist, newcameramtx, fx, x_cen_cor, offset

    #----------------------------------------------------------------------------
    # Euclidian afstand van pixels tot NUL-pixels bepalen
    distance = cv.distanceTransform(eroded_voor_diepte, cv.DIST_L2, 5)

    # Rand voor splitsen L/R
    rand = int(offset * 2)

    dist_links = distance[:, :rand]
    dist_rechts = distance[:, rand:]

    # Gemaakt plaatje tekenbaar door opencv maken 
    dist_output = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)

    # cv.imshow('eroded_voor_diepte', eroded_voor_diepte)
    # cv.imshow('dist_output', dist_output)

    # Zoek de pixel met de grootste afstand
    _, max_radius_links, _, center_links = cv.minMaxLoc(dist_links)
    _, max_radius_rechts, _, center_rechts = cv.minMaxLoc(dist_rechts)

    # Middelpunt en straal
    x_cen_links, y_cen_links = center_links
    x_cen_rechts, y_cen_rechts = center_rechts

    x_cen_rechts += rand

    # Correctie voor andere framemaat 

    radius_links = max_radius_links
    radius_rechts = max_radius_rechts

    print("Delta test: ", int(x_cen_links) - int(x_cen_rechts), "\n")

    afstand_Za, afstand_Zb = get_distance(x_cen_links, x_cen_rechts, hoogte_camera)
    #----------------------------------------------------------------------------

    # Gewilde variabelen terugleveren 
    #----------------------------------------------------------------------------
    # return x_cen_links, x_cen_rechts, radius_links, radius_rechts, troubleshoot_euc
    return afstand_Za, afstand_Zb, radius_links, radius_rechts, dist_output, x_cen_links, x_cen_rechts
    #----------------------------------------------------------------------------

# Functie voor het bepalen van de afstand dmv witte punten 
def wittepunten_afstand(gradient_diepte, hoogte_camera): 
    global x_cen_cor, x_lin_cor, offset

    # Rand voor splitten L/R
    rand = int(offset * 2)

    # Witte pixels bijhouden 
    x_wit_rechts = np.empty(0, dtype = int) 
    x_wit_links = np.empty(0, dtype = int) 

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
            if (x_values[l] > rand): 
                x_wit_rechts = np.append(x_wit_rechts, (x_values[l] - 0))
            else: 
                x_wit_links = np.append(x_wit_links, (x_values[l] - 0))

    # For loop geeft per gevonden pixelpaar een output, dit moet gemiddeld worden 
    gem_x_l = np.mean(x_wit_links) 
    gem_x_r = np.mean(x_wit_rechts)

    # Maximale en minimale x coordinaten per foto 
    x_l_l = np.min(x_wit_links)
    x_l_r = np.max(x_wit_links)
    x_r_l = np.min(x_wit_rechts)
    x_r_r = np.max(x_wit_rechts)

    print("\ncoordinaten rechts: ", gem_x_r)
    print("coordinaten links: ", gem_x_l)
    print("Delta tussen coordinaten (Pixels): ", int(gem_x_l - gem_x_r))

    # Berekenen van de afstand met de gevonden coordinaten 
    lengteZa_witL, lengteZb_witL = get_distance(int(x_l_l), int(x_r_l), hoogte_camera)
    lengteZa_witR, lengteZb_witR = get_distance(int(x_l_r), int(x_r_r), hoogte_camera)

    lengteZa_wit, lengteZb_wit = get_distance(int(gem_x_l), int(gem_x_r), hoogte_camera)

    # return x_l_l, x_r_l, x_l_r, x_r_r
    return lengteZa_witL, lengteZb_witL, lengteZa_witR, lengteZb_witR, gem_x_l, gem_x_r, lengteZa_wit, lengteZb_wit
    # return x_l_l, x_l_r, x_r_l, x_r_r, gem_x_l, gem_x_r, lengteZa_wit, lengteZb_wit

# Splitten van de dubbele frame voor gemak van verwerking 
def fotosplitter(frame, positie, afstand): 
    global x_cen_cor, x_lin_cor, offset, mtx, dist, newcameramtx
    # x_lin_cor = 30, delta = 716 
    # x_cen_cor = 746, delta = 534 

    x_lin_spoof = 212
    
    frame_width = 1280
    frame_height = 720

    if frame is None or frame.size == 0: 
        print("Fout in frame")
        return None, None 

    # Splitten van frame dmv ingebouwde functie in openCV
    linker_frame = frame[0: frame_height, x_lin_spoof:x_cen_cor]
    rechter_frame = frame[0: frame_height, x_cen_cor:frame_width]

    # Posities bounding box bepalen (links)# --- LINKER FRAME UITSNIJDEN ---
    y_1_l, y_2_l = hoeken_vinden(positie[0], 1, 0, afstand)
    x_1_l, x_2_l = hoeken_vinden(positie[1], 0, 0, afstand)

    h_l, w_l = linker_frame.shape[:2]
    center_y_l, center_x_l = int(h_l / 2), int(w_l / 1.7)

    # Middelpunt meerekenen voor links
    y_1_l += center_y_l
    y_2_l += center_y_l
    x_1_l += center_x_l
    x_2_l += center_x_l

    # Veiligheidsschek en clipping (links)
    y_start_l = max(0, min(min(int(y_1_l), int(y_2_l)), h_l))
    y_end_l   = max(0, min(max(int(y_1_l), int(y_2_l)), h_l))
    x_start_l = max(0, min(min(int(x_1_l), int(x_2_l)), w_l))
    x_end_l   = max(0, min(max(int(x_1_l), int(x_2_l)), w_l))

    klein_links = linker_frame[y_start_l:y_end_l, x_start_l:x_end_l]

    # --- RECHTER FRAME UITSNIJDEN ---
    y_1_r, y_2_r = hoeken_vinden(positie[0], 1, 0, afstand)
    x_1_r, x_2_r = hoeken_vinden(positie[1], 0, 0, afstand)

    h_r, w_r = rechter_frame.shape[:2]
    center_y_r, center_x_r = int(h_r / 2), int(w_r / 2.3)

    # Middelpunt meerekenen voor rechts (met de andere X-factor)
    y_1_r += center_y_r
    y_2_r += center_y_r
    x_1_r += center_x_r
    x_2_r += center_x_r

    # Veiligheidsschek en clipping (rechts)
    y_start_r = max(0, min(min(int(y_1_r), int(y_2_r)), h_r))
    y_end_r   = max(0, min(max(int(y_1_r), int(y_2_r)), h_r))
    x_start_r = max(0, min(min(int(x_1_r), int(x_2_r)), w_r))
    x_end_r   = max(0, min(max(int(x_1_r), int(x_2_r)), w_r))

    klein_rechts = rechter_frame[y_start_r:y_end_r, x_start_r:x_end_r]

    # Grijs maken van de frames, wilt een functie later graag 
    linker_frame = cv.cvtColor(linker_frame, cv.COLOR_BGR2GRAY)
    rechter_frame =  cv.cvtColor(rechter_frame, cv.COLOR_BGR2GRAY)

    # Samenvoegen van de verkleinde plaatjes 
    klein_combi = np.hstack((klein_links, klein_rechts))

    return linker_frame, rechter_frame, klein_links, klein_rechts, klein_combi

# Functie voor het omzetten van een 3d-vector naar de afstand, hierin zit TCP instelling vermingeld 
# De transformatie vanuit de robot is een vector van het nulpunt van de arm, deze moet dus bewerkt worden 
def afstand_uit_3d_trans(transformatie, rotatie): 
    # Voor het aanpassen van de matrix, de rvec en tvec van de gekozen feature 
    # Rotatie is radialen 
    feature_rotatievector = [3.112451989161527, 0.012513923491795566, -0.01010592710831033]

    # Transformatie in meter 
    feature_transformatievector = [0.03331083849040679, -0.7581268709319913, -0.6871726072217264]

    # Voor feature definitie 
    # -----------------------------------------------------------------------------------------
    # Rotatiematrix opstellen, deze is nodig voor het omvormen van de transformatie 
    feature_rotatie_matrix = R.from_rotvec(feature_rotatievector).as_matrix()

    # Lege 4x4 matrix opstellen, een zogehete identity matrix 
    feature_t_matrix = np.eye(4)

    # Matrix vullen 
    feature_t_matrix[:3, :3] = feature_rotatie_matrix
    feature_t_matrix[:3, 3] = feature_transformatievector
    # -----------------------------------------------------------------------------------------

    # Actuele data 
    # -----------------------------------------------------------------------------------------
    # Rotatiematrix opstellen, deze is nodig voor het omvormen van de transformatie 
    rotatie_matrix = R.from_rotvec(rotatie).as_matrix()

    # Lege 4x4 matrix opstellen, een zogehete identity matrix 
    t_matrix = np.eye(4)

    # Matrix vullen 
    t_matrix[:3, :3] = rotatie_matrix
    t_matrix[:3, 3] = transformatie
    # -----------------------------------------------------------------------------------------

    # Deze matrix is nu vanuit de base, waar de arm staat, maar hij moet vanuit de feature zijn, waar het object staat 
    t_feature_tcp = np.linalg.inv(t_matrix) @ feature_t_matrix

    transformatie_tov_feature = t_feature_tcp[:3, 3]
    # print("transformatie_tov_feature:", transformatie_tov_feature)

    afstand = np.sqrt(((transformatie_tov_feature[0] * 1000) ** 2) + ((transformatie_tov_feature[1] * 1000) ** 2) + ((transformatie_tov_feature[2] * 1000) ** 2))
    # print("afstand tot nulpunt (TCP):", afstand)

    afstand += 175
    # print("afstand tot nulpunt (Lenzen):", afstand, "\n")

    return afstand 

# Tussen de twee afstandsmethodes gewogen avg, zodat een hogere afwijking minder telt 
def gewogen_avg(inputs):
    verwacht = 100.0 
    tolerantie = 15.0 

    afwijking = np.abs(inputs - verwacht)
    gewichten = np.exp(-(afwijking**2) / (2 * (tolerantie**2)))

    resultaat = np.average(inputs, weights=gewichten)
    return resultaat

# Het maken van een bounding box op een geometrisch correcte wijze 
def hoeken_vinden(rotatie_graden, centerpunt, centered, afstand): 
    global fx, fy
    # fx = 1008.33
    # fy = 1001.35
    
    S_tcp = 50 # In mm
    S_tcp += afstand 

    Breedte_tafel = 200 # In mm

    if centerpunt == 1: 
        Center = 360
        fb = fy 
        # print("Centerpunt feedback: je hebt fy gekozen!")
    else:
        Center = 640
        fb = fx
        # print("Centerpunt feedback: je hebt fx gekozen!")

    DB = Breedte_tafel / 2 

    # theta = np.deg2rad(90 - np.abs((rotatie_graden - 180)) % 180)
    theta = np.deg2rad(rotatie_graden - 180)

    Xa = -1 * (Breedte_tafel / 2) * np.cos(theta)
    Za = S_tcp - (Breedte_tafel / 2) * np.sin(theta)

    Xb = (Breedte_tafel / 2) * np.cos(theta)
    Zb = S_tcp + (Breedte_tafel / 2) * np.sin(theta)

    if centered == 1: 
        x_1 = Center + (fb * Xa/Za)
        x_2 = Center + (fb * Xb/Zb)

    else: 
        x_1 =  (fb * Xa/Za)
        x_2 =  (fb * Xb/Zb)

    return x_1, x_2 

# Functie voor het extraheren van lijnen, voor de hendels 
def Find_lines(gradient_magnitude):
    # ---------------- HOUGH LINES ----------------
    avg_lengte = []
    Hoek_array = []
    
    #Hough lines Propabilistic 
    linesP = cv.HoughLinesP(gradient_magnitude, 1, np.pi / 180, 83, None, 50, 15)

    if linesP is not None:
        for l in linesP:
            l = l.flatten()
            x1, y1, x2, y2 = map(int, l)
            #als hij minder dan 2 lines ziet niet meenemen dan ziet hij strays
            if(len(linesP) > 2):
                lengte = math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
                hoek = math.acos(abs(x1-x2)/lengte)
                avg_lengte.append(lengte)
                Hoek_array.append(hoek)
                # Hoek_lengte_array.append((lengte, hoek))

        #kijk of de lengte van de lengte-/hoek_array 0 is anders avg pakken.
        if((len(avg_lengte) != 0) or (len(Hoek_array) != 0)):   #hier keurt hij 1 lijn af.
            avg_lengte_out = sum(avg_lengte)/len(avg_lengte)  
            
            Hoek_overeenkomend = []
            gem_hoek = (sum(Hoek_array)/len(Hoek_array))
            
            for Index_hoek in range(len(Hoek_array)):
                # radialen YIPPIE :(
                abs_hoek = abs((gem_hoek)-Hoek_array[Index_hoek])
                if(abs_hoek < 0.07 ):#deze aanpassen als je strenger wilt filteren of een hoek in de buurt zit van de anderen.
                    Hoek_overeenkomend.append(Hoek_array[Index_hoek])
                    
            if(len(Hoek_overeenkomend) >= 2):
                return (avg_lengte_out) #hier zetten wat hij moet krijgen om te zeggen ja ik heb een steelpan/koekepan
    return (0)

# Dictionaries voor het omzetten van nummer, waar KNN in getraint wordt, naar label, dat wij kunnen lezen 
#----------------------------------------------------------------------------------------------
unieke_labels = ['cor_mok_kookpan_middel', 
                 'cor_koekenpan', 
                 'cor_kookpan', 
                 'cor_mok', 
                 'cor_kookpan_klein', 
                 'cor_steelpan', 
                 'cor_glazenbeker', 
                 'cor_mok_kookpan_middelgroot', 
                 'cor_mok_groot']

label_naar_nummer = {}

for i, label in enumerate(unieke_labels):
    label_naar_nummer[label] = i

nummer_naar_label = {}

for label, nummer in label_naar_nummer.items():
    nummer_naar_label[nummer] = label

# Beginnen main functieloop     
def main():
    global laatste_robot_bericht, Teller_foto, foto_trigger, go, mtx, dist, newcameramtx, offset

    # Teller voor vernieuwde dataset 
    counter = 0

    # Schrijven Posities
    teller = 0

    # Waarden voor het normaliseren van de radius mashup
    normalisatie_min = 5.62319310
    normalisatie_max = 36.95314243

    # Waarden voor het normaliseren van de afstand 
    normalisatie_min_S = 186.61373685
    normalisatie_max_S = 135.01729479

    # Timen van de foto nemen 
    starttijd = None 
    wachttijd = 2.5
    opslaan = None

    # Informatie van de server vaststellen 
    HOST = "192.168.106.199"
    PORT = 5000

    # IP-adres van de robotarm, ingesteld op gegeven controller 
    IP_ADRES = "192.168.106.159"

    # Asynchrone server starten, dit moet ook op een aparte thread, vanwege constant onderhoud dat vereist is 
    socket_thread = threading.Thread(target=robot_connectie, args=(HOST, PORT), daemon=True)
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
    Radius_euc = np.empty(0, dtype = np.float32) 

    # Variabelen voor het tracken van afstand dmv methode wittepuntendetectie 
    Zalist_wit = np.empty(0, dtype = int) 
    Zblist_wit = np.empty(0, dtype = int) 

    Zab_totaal_tussen = np.empty(0, dtype = int)
    Zab_totaal = np.empty(0, dtype = int)

    lijn_results = np.empty(0, dtype = np.float32)

    # Window aanmaken op correcte maat, Wayland dingetje 
    cv.namedWindow('Lookin', cv.WINDOW_NORMAL)
    cv.resizeWindow('Lookin', int(frame_width), int(frame_height))

    # Feedback naar terminal printen 
    print(f"Server en Videoconnectie gestart!")

    # RTSP stream van de camera inlezen
    url = "rtsp://admin:admin123@192.168.106.149:554/line=2/stream=1" 
    cam = cv.VideoCapture(url)

    # Aanmaken van de KNN
    KNN = cv.ml.KNearest_create()

    # Voorarray voor KNN aanmaken
    KNN_voer = np.empty((1, 3), dtype = np.float32)
    
    # KNN trainen 
    gevormde_data = np.load("KNN_trainset.npy")
    labels = np.load("KNN_labels.npy")

    KNN.train(gevormde_data,
              cv.ml.ROW_SAMPLE,
              labels
              )
    
    # Main loop
    try:
        while True:
            # Camera uitlezen begin van elke loop 
            ret, frame = cam.read()
            frame_copy = frame.copy()

            # Framecorrectie uitvoeren
            frame_cor = cv.undistort(frame, mtx, dist, None, newcameramtx)

            # Actuele positie van de robot uitlezen via RTDE
            actuele_positie = rtde_r.getActualTCPPose()

            # Alleen transformatie pakken 
            transformatievector = actuele_positie[:3]              

            # Enkel de rotatie pakken 
            rotatievector = actuele_positie[3:]
            rotatievector_graden = R.from_rotvec(rotatievector).as_euler('xyz', degrees=True)

            #Cancellen als de camera niet werkt
            if frame is not None: 
                # Bepalen van de afstand uit gegeven vectoren van armen 
                d = afstand_uit_3d_trans(transformatievector, rotatievector)
                print(d)

                # Stereo afstand bepalen voor zoning, deze is exclusief aan deze samenvoeging 
                Lf, Rf, _, _, imgCK = fotosplitter(frame_cor, rotatievector_graden, d)   
                hg, offset = imgCK.shape[:2]  
                offset = offset / 4 

                cv.imshow('Goofy', imgCK)
                cv.imshow('Lookin', frame)

            # Verplichte exit strategie uit het opencv window
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Gestopt via window")
                break

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
                # Programma stoppen als 10 is bereikt, tijdelijk voor testen   
                if teller == 10: 
                    print("Gem(hoogte):", np.mean(Zab_totaal))
                    print("Gem(radius):", np.mean(Radius_euc))
                    print("Gem(lijnen):", np.mean(lijn_results))

                    KNN_voer[0][1] = np.mean(Radius_euc)
                    KNN_voer[0][0] = np.mean(Zab_totaal)

                    ret, resultaat, buren, afstanden = KNN.findNearest(
                        KNN_voer,
                        k=5
                    )
    
                    # if resultaat[0][0] < 7:
                    print("Voorspeld label:", nummer_naar_label[int(resultaat[0][0])])

                    while(1): 
                        print("\nStop het programma!")
                        time.sleep(2)

                cv.imwrite(f"./testfotos/foto_positie_{Teller_foto}.png", imgCK)
                print(f"\nFoto gemaakt({Teller_foto})!\n")

                # counter += 1    
                # cv.imwrite(f"./saved_uncor/mok{counter}.png", frame)
                # cv.imwrite(f"./saved_cor/mok{counter}.png", frame_cor)
                # print("Image sizes: ", frame_width, " - ", frame_height)

                # print('\ntransformatie', transformatievector)
                # print('rotatievector', rotatievector)
                # print('rotatievector_graden', rotatievector_graden)

                # Frame door de pre-processing gooien
                processed_frame, _, pre_euc_doorgeef, processed_lijn = preprocessing(imgCK)
                # cv.imshow('Klaar?', processed_frame)
                # cv.imshow('Klaar', pre_euc_doorgeef)
                
                # Huidige hoogte berekenen, deze is benodigd voor alle afstandsfunctie 
                # d = afstand_uit_3d_trans(transformatievector, rotatievector)

                lijn_result = Find_lines(processed_lijn)
                lijn_results = np.append(lijn_results, lijn_result)
                print("test_1", lijn_results)

                # Eerste afstandsmethode toepassen 
                a, b, c1, c2, _, cl, cr = euclidian_afstand(pre_euc_doorgeef, d) 

                # cv.circle(frame_copy, (int(a), 360), 2, (0, 0, 255), 3)
                # cv.circle(frame_copy, (int(b), 360), 2, (0, 0, 255), 3)
                # cv.imshow('midden_cirkels', frame_copy)

                # Berekende waarden opslaan 
                Zalist_euc = np.append(Zalist_euc, int(a))
                Zblist_euc = np.append(Zblist_euc, int(b))

                print("rad1", c1)
                print("rad2", c2)

                # Voordat de radius gebruikt kan worden, moet deze met afstand vermingeld worden 
                Radius_final = (d / ((c1 + c2) / 2))

                Radius_final = (Radius_final - normalisatie_min) / ((normalisatie_max/2) - normalisatie_min)

                Radius_euc = np.append(Radius_euc, Radius_final)

                print('Zalist_euc', Zalist_euc)
                # print('Zblist_euc', Zblist_euc)
                print('Radius_euc', Radius_euc)

                # Tweede afstandsmethode toepassen 
                l1, l2, r1, r2, clw, crw, ga, gb = wittepunten_afstand(processed_frame, d)

                rand = int(offset * 2)
                cv.line(imgCK, (rand, 0), (rand, rand), (255, 0, 0), 3)

                # Voor euc
                cv.circle(imgCK, (int(cl), 185), 5, (0, 255, 0), 3)
                cv.circle(imgCK, (int(cr), 185), 5, (0, 255, 0), 3)

                # Voor witte gem
                cv.circle(imgCK, (int(clw), 185), 5, (0, 0, 255), 3)
                cv.circle(imgCK, (int(crw), 185), 5, (0, 0, 255), 3)

                cv.imshow('Goofeh', imgCK)

                # Gemiddelden van de teruggekregen waarden pakken 
                l = int((l1 + r1) / 2)
                r = int((l2 + r2) / 2)

                # Gemiddeld opslaan voor plotten 
                Zalist_wit = np.append(Zalist_wit, ga)
                Zblist_wit = np.append(Zblist_wit, gb)

                # Feedback aan gebruiker 
                print('Zalist_wit', Zalist_wit)
                # print('Zblist_wit', Zblist_wit)           
                #     
                Zab_totaal_tussen = np.append(Zab_totaal_tussen, [[int(ga)], [int(a)]])

                zijstap = gewogen_avg(Zab_totaal_tussen)
                zijstap = (zijstap - normalisatie_min_S) / ((normalisatie_max_S) - normalisatie_min_S)

                Zab_totaal = np.append(Zab_totaal, zijstap)

                print('Zab_totaal_tussen', Zab_totaal_tussen)
                print('zab_totaal', Zab_totaal)

                KNN_voer[0][2] = lijn_result
                KNN_voer[0][1] = Radius_final
                KNN_voer[0][0] = zijstap

                # KNN_voer[0][1] = 14.0
                # KNN_voer[0][0] = 100.0

                print("KNN_voer", KNN_voer)

                ret, resultaat, buren, afstanden = KNN.findNearest(
                    KNN_voer,
                    k=5
                )

                # if resultaat[0][0] < 7:
                print("Voorspeld label:", nummer_naar_label[int(resultaat[0][0])])

                # Opslaan van alle benodigde variabelen
                teller += 1
                with open("test_resultaten.txt", "a") as file:
                    file.write(f"Positie en oriëntatie van de TCP in positie {teller}: {transformatievector} / {rotatievector_graden}\n")
                print(f"Positie en oriëntatie van de TCP in positie {teller} opgeslagen in test_resultaten.txt")

                # Legen van tijdelijke array
                Zab_totaal_tussen = np.empty(0, dtype=np.float32)
                # KNN_voer = np.delete(Zab_totaal_tussen, (len(KNN_voer) - 1))

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

# Main functie laten lopen
if __name__ == "__main__":
    main()