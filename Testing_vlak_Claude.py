import ast
import glob
import os
import re

import cv2
import numpy as np

# Eerder gegenereerde correctiedata voor camera inladen
correction_data = np.load("camera_calibration.npz")
mtx = correction_data["camera_matrix"]
DIST = correction_data["dist_coeffs"]
K = correction_data["newcameramtx"]

# ----------------------------------------------------------------------
# INSTELLINGEN (aanpassen)
# ----------------------------------------------------------------------
POSE_BESTAND = "test_resultaten_PERM.txt"
FOTO_MAP = "testfotos"                 # map met de plaatjes
FOTO_PATROON = "*.png"             # of *.jpg
UITVOER_MAP = "uitvoer"            # hier komen gemaskeerde beelden

# Camera t.o.v. TCP
AFSTAND_TCP_CAMERA = 0.20          # m, camera zit 20 cm achter de TCP
BASELINE = 0.02                    # m, afstand tussen de twee camera's (invullen!)
LINKER_CAMERA = True               # True: linkercamera = -baseline/2 t.o.v. cameramidden

# Platform in het platformframe (m), nulpunt in het midden, z = 0 op het platformvlak
PLATFORM = np.array([[-0.15, -0.10, 0.0],
                     [ 0.15, -0.10, 0.0],
                     [ 0.15,  0.10, 0.0],
                     [-0.15,  0.10, 0.0]])

# Hoogte van het object boven het platform (m). 0 = alleen het platformvlak.
# Let op de richting van je z-as: pas het teken aan als het masker de verkeerde kant op uitsteekt.
OBJECT_HOOGTE = 0.0

# ----------------------------------------------------------------------
# Hulpfuncties
# ----------------------------------------------------------------------
def lees_poses(pad):
    """Leest per regel de lijst [x,y,z,rx,ry,rz] uit het tekstbestand."""
    poses = []
    with open(pad, "r") as f:
        for line in f:
            if "[" in line and "]" in line:
                start = line.find("[")
                eind = line.rfind("]") + 1
                getallen = ast.literal_eval(line[start:eind])
                poses.append(getallen)
    data = np.array(poses, dtype=np.float64)   # float64 voor OpenCV-berekeningen
    return data[:, :3], data[:, 3:]            # tvec, rvec


def natuurlijk_sorteren(pad):
    """Sorteert foto_2 vóór foto_10 (anders staat foto_10 na foto_1)."""
    naam = os.path.basename(pad)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", naam)]


def pose_naar_T(tvec, rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def inverteer_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def maak_T_tcp_cam():
    """Camera-frame uitgedrukt in het TCP-frame (alleen translatie, geen rotatie)."""
    T = np.eye(4)
    x_offset = -BASELINE / 2 if LINKER_CAMERA else 0.0
    T[:3, 3] = [x_offset, 0.0, -AFSTAND_TCP_CAMERA]
    return T


def projecteer_platform(tvec, rvec, T_tcp_cam):
    """Geeft de platformhoeken (en evt. bovenvlak) als 2D-pixels voor één pose."""
    T_platform_tcp = pose_naar_T(tvec, rvec)
    T_platform_cam = T_platform_tcp @ T_tcp_cam
    T_cam_platform = inverteer_T(T_platform_cam)

    rv, _ = cv2.Rodrigues(T_cam_platform[:3, :3])
    tv = T_cam_platform[:3, 3].reshape(3, 1)

    punten = PLATFORM.copy()
    if OBJECT_HOOGTE != 0:
        boven = PLATFORM.copy()
        boven[:, 2] += OBJECT_HOOGTE
        punten = np.vstack([punten, boven])

    # Controle: alle punten moeten voor de camera liggen
    pts_cam = (T_cam_platform[:3, :3] @ punten.T + T_cam_platform[:3, 3:4]).T
    voor_camera = bool(np.all(pts_cam[:, 2] > 0))

    px, _ = cv2.projectPoints(punten, rv, tv, K, DIST)
    px = px.reshape(-1, 2)

    if OBJECT_HOOGTE != 0:
        px = cv2.convexHull(px.astype(np.float32)).reshape(-1, 2)
    return px, voor_camera


# ----------------------------------------------------------------------
# Hoofdprogramma
# ----------------------------------------------------------------------
def main():
    tvecs, rvecs = lees_poses(POSE_BESTAND)
    fotos = sorted(glob.glob(os.path.join(FOTO_MAP, FOTO_PATROON)), key=natuurlijk_sorteren)

    print(f"{len(tvecs)} poses, {len(fotos)} foto's")
    if len(tvecs) != len(fotos):
        raise SystemExit("Aantal poses en foto's komt niet overeen, controleer de volgorde/inhoud.")

    os.makedirs(UITVOER_MAP, exist_ok=True)
    T_tcp_cam = maak_T_tcp_cam()

    for i, (pad, tvec, rvec) in enumerate(zip(fotos, tvecs, rvecs)):
        img = cv2.imread(pad)
        if img is None:
            print(f"Kan {pad} niet lezen, overgeslagen")
            continue

        px, voor_camera = projecteer_platform(tvec, rvec, T_tcp_cam)
        if not voor_camera:
            print(f"[{i}] {os.path.basename(pad)}: platform (deels) achter de camera, controleer pose/frames")

        # Masker maken en toepassen
        masker = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(masker, [np.round(px).astype(np.int32)], 255)
        roi = cv2.bitwise_and(img, img, mask=masker)

        # >>> hier je eigen preprocessing op 'roi' of met 'masker' <<<

        # Visuele controle: omlijning op origineel
        controle = img.copy()
        cv2.polylines(controle, [np.round(px).astype(np.int32)], True, (0, 255, 0), 2)

        naam = os.path.splitext(os.path.basename(pad))[0]
        cv2.imwrite(os.path.join(UITVOER_MAP, f"{naam}_controle.png"), controle)
        cv2.imwrite(os.path.join(UITVOER_MAP, f"{naam}_roi.png"), roi)

        cv2.imshow("controle (q = stop)", controle)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()