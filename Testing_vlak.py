from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
import numpy as np
import ast
import cv2 as cv
import re
import glob

# Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
dist = correction_data["dist_coeffs"]
newcameramtx = correction_data["newcameramtx"]

print("newcameramtx", newcameramtx)
print("dist", dist)
print("mtx", mtx)

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
fy = mtx[1, 1]
print("fy", fy)
# ------------------------------------------------------------------------------------------------------

# Te testen frames inladen 
Testfotos = glob.glob('testfotos/*.png') 

# Te vergelijken vlak opmaken, moet nog getuned worden 
deviatie = 0.05
basis_vlak = np.array( 
    [ 
        [-deviatie, -deviatie, 0.0],  # Linksboven
        [deviatie, -deviatie, 0.0],  # Rechtsboven
        [deviatie, deviatie, 0.0],  # Rechtsonder
        [-deviatie, deviatie, 0.0],  # Linksonder
    ],
    dtype = np.float32
)

# Posities uit opgeslagen lijst transleren 
alle_posities = []

# Opgeslagen lijst uitlezen 
with open("test_resultaten_PERM.txt", "r") as file:
    for line in file:
        if "[" in line and "]" in line:

            # Begin en eindindexen localiseren
            start_index = line.find("[")
            eind_index = line.rfind("]") + 1
            
            lijst_als_tekst = line[start_index:eind_index]
            getallen_lijst = ast.literal_eval(lijst_als_tekst)

            alle_posities.append(getallen_lijst)
        print("alle_posities", alle_posities)

    data_matrix = np.array(alle_posities, dtype = np.float32)

    tvec = data_matrix[:, :3]
    rvec = data_matrix[:, 3:]

    print("\nHet zijn er: ", len(alle_posities))

def ur_pose_naar_opencv(tvec_ur, rvec_ur):
  # 1. Maak van de UR rvec en tvec een 4x4 transformatiematrix (T_feature_tcp)
  rot_matrix = R.from_rotvec(rvec_ur).as_matrix()
  T_feature_tcp = np.eye(4)
  T_feature_tcp[:3, :3] = rot_matrix
  T_feature_tcp[:3, 3] = tvec_ur

  # 2. INVERTEREN: OpenCV verwacht de transformatie van Vlak -> Camera
  # De robot geeft TCP ten opzichte van Vlak, dus we moeten dit omdraaien.
  T_cam_feature = np.linalg.inv(T_feature_tcp)

  # 3. Optionele correctie (als je merkt dat het beeld op de kop staat of spiegelt)
  # Mocht de camera bijvoorbeeld 180 graden gedraaid hangen, kun je hier een correctie-rotatie toevoegen:
  # R_correction = R.from_euler('x', 180, degrees=True).as_matrix()
  # T_cam_feature[:3, :3] = R_correction @ T_cam_feature[:3, :3]

  # 4. Omzetten van de 4x4 matrix terug naar OpenCV rvec en tvec
  rvec_cv = R.from_matrix(T_cam_feature[:3, :3]).as_rotvec()
  tvec_cv = T_cam_feature[:3, 3]

  return rvec_cv, tvec_cv

def extract_number(filepath):
  numbers = re.findall(r'\d+', filepath)
  return int(numbers[-1]) if numbers else 0

rot_grad = np.empty((len(alle_posities), 3), dtype = np.float32) 

def maak_graden(rvec):
    for i in range(len(rvec)): 
        # Array aanmaken voor behouden hoeken 
        # 1. Converteer rotatievector naar rotatiematrix
        rot = R.from_rotvec(rvec[i])

        # 2. Haal de losse hoeken op om de X, Y en Z-as (in graden)
        # 'xyz' betekent hier de extrinsieke rotatievolgorde (Roll om X, Pitch om Y, Yaw om Z)
        rpy_graden = rot.as_euler('xyz', degrees=True)
        rot_grad[i] = rpy_graden

        print(f"\n\nRvec nummer: {i}")
        print(f"Rotatie om X-as (Roll):  {rpy_graden[0]:.2f}°")
        print(f"Rotatie om Y-as (Pitch): {rpy_graden[1]:.2f}°")
        print(f"Rotatie om Z-as (Yaw):   {rpy_graden[2]:.2f}°")

    return rot_grad

def hoeken_vinden(rotatie_graden): 
    global fx, fy

    S_tcp = 705 # In mm - Dubbel
    Breedte_tafel = 800 # In mm
    DB = Breedte_tafel / 2 

    theta_1 = np.deg2rad(90 - (rotatie_graden[0] - 90))
    DE = (np.cos(theta_1)) * S_tcp
    CE = np.sin(theta_1) * S_tcp
    BE = np.abs(DE - DB)
    AE = DE + DB

    BC = np.sqrt(CE ** 2 + BE ** 2)
    AC = np.sqrt(CE ** 2 + AE ** 2)

    alpha = np.arccos(((BC ** 2) + (AC ** 2) - (Breedte_tafel ** 2)) / (2 * BC * AC))
    alpha_1 = np.arccos(((BC ** 2) + (S_tcp ** 2) - (DB ** 2)) / (2 * BC * S_tcp))
    alpha_2 = np.arccos(((AC ** 2) + (S_tcp ** 2) - (DB ** 2)) / (2 * AC * S_tcp))
    
    print("alpha:", np.rad2deg(alpha)) 
    print("alpha_1:", np.rad2deg(alpha_1)) 
    print("alpha_2:", np.rad2deg(alpha_2)) 

    x_1 = np.tan(alpha_1) * fx 
    x_2 = np.tan(alpha_2) * fx 

    return x_1, x_2 

def hoeken_vinden_test(rotatie_graden): 
    global fx, fy
    # fx = 1008.33
    # fy = 1001.35
    
    S_tcp = 235 # In mm - Dubbel
    Breedte_tafel = 150 # In mm
    Cx = 360
    DB = Breedte_tafel / 2 

    # theta = np.deg2rad(90 - np.abs((rotatie_graden - 180)) % 180)
    theta = np.deg2rad(rotatie_graden - 180)

    Xa = -1 * (Breedte_tafel / 2) * np.cos(theta)
    Za = S_tcp - (Breedte_tafel / 2) * np.sin(theta)

    Xb = (Breedte_tafel / 2) * np.cos(theta)
    Zb = S_tcp + (Breedte_tafel / 2) * np.sin(theta)

    x_1 = Cx + (fy * Xa/Za)
    x_2 = Cx + (fy * Xb/Zb)

    return x_1, x_2 

# Sorteer de lijst strikt op numerieke volgorde
gestordeerde_fotos = sorted(Testfotos, key=extract_number)

# Bijhouden welke vector te benutten 
vector_teller = 0 

for i, fname in enumerate(gestordeerde_fotos): 
    img = cv.imread(fname) 

    rvec_opencv, tvec_opencv = ur_pose_naar_opencv(tvec[vector_teller], rvec[vector_teller])

    print("\ntvec:", tvec[vector_teller])
    print("\nrvec:", rvec[vector_teller])

    rotatie = maak_graden(rvec)
    x_1, x_2 = hoeken_vinden(rotatie[i])   

    h_l, w_l = img.shape[:2] 
    print("h_l, w_l = linker_frame.shape[:2]", h_l, w_l)
    x_1_t, x_2_t = hoeken_vinden_test(rotatie[i][0]) 
    
    print("\nHij gebruikt deze foto:", fname)
    print("Dit zijn de pixels:", x_1_t, x_2_t)
    print("Dit is de draaihoek ten opzichte van hoog:", ((rotatie[i][0] - 90) % 360), "\n")

    # cv.circle(img, (int(x_1), int(y_cen)), 2, (0, 255, 255), 3)

    cv.circle(img, (int((frame_width)/2), int(x_1_t)), radius=10, color=(0, 255, 0), thickness=2)
    cv.circle(img, (int((frame_width)/2), int(x_2_t)), radius=10, color=(255, 0, 0), thickness=2)

    # 2. Projecteer direct naar 2D pixels (OpenCV berekent de 'driehoeken' onder hood!)
    # (Zorg dat rvec_cv en tvec_cv de omegerekende vectoren zijn zoals eerder besproken)
    pixels_2d, _ = cv.projectPoints(
        basis_vlak, rvec_opencv, tvec_opencv, newcameramtx, dist
    )

    pixels_2d = np.int32(pixels_2d).reshape(-1, 2)

    # 3. Teken direct het vlak op je beeld
    cv.polylines(img, [pixels_2d], isClosed=True, color=(0, 255, 0), thickness=2)

    print("pixels_2d", pixels_2d)
    
    cv.imshow("Test", img)

    vector_teller += 1 
    cv.waitKey(0)

# Bewaren! 
# [-0.0038241850493679615, -0.5565176157764261, 0.07711386747081989, 2.3499750673187907, 0.004248655653312149, -0.0013394303516092547]