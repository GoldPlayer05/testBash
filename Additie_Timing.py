import rtde_receive
from rtde_receive import RTDEReceiveInterface
import rtde_control 

import ast
import cv2 as cv
import socket
import threading 
import time

# Laatste bericht robot bijhouden, moet globaal vanwege Async functionaliteit 
laatste_robot_bericht = ""

# Data veilig delen tussen threads 
data_lock = threading.Lock()

# Variabele om te controleren of er een foto moet worden genomen
foto_trigger = False 

# Gaan voor volgende positie 
go = False 

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
                # input("\nDruk op [ENTER] in je terminal om de robot groen licht te geven")

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