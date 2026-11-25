import numpy as np
from scipy.spatial.transform import Rotation as R
import ast

# Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
dist = correction_data["dist_coeffs"]
newcameramtx = correction_data["newcameramtx"]

# Voor de berekening van afstand vocale lengte camera uit bestand halen 
fx = mtx[0, 0] 
fy = mtx[0, 1]

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

    data_matrix = np.array(alle_posities, dtype = np.float32)

    tvec = data_matrix[:, :3]
    rvec = data_matrix[:, 3:]

    print("\nHet zijn er: ", len(alle_posities))

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
    global fx 

    S_tcp = 235 # In mm 
    Breedte_tafel = 400 # In mm
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

rotatie = maak_graden(rvec)

x_1, x_2 = hoeken_vinden(rotatie[0])

print("\nDit zijn de coordinaten: ", x_1, x_2)
print("\nFinal score: \n", rotatie)