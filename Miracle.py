"""
Walkthrough_Prog_stap3-rework_v2.py

Maakt de KNN-trainset uit opgeslagen foto's met EXACT dezelfde detectieketen
als het live-programma (Additie_bounding-box.py):

    fotosplitter (bounding box) -> preprocessing -> Find_lines
    -> euclidische afstand + witte punten -> gewogen gemiddelde

Verschillen met live die hier worden opgevangen:
  * Foto's zijn genomen op 500 mm, live draait op 325 mm. Daarom:
      - de bounding box wordt met afstand=500 berekend (zelfde fysieke tafel),
      - cx (optisch midden in de crop) en de pixel-parameters schalen mee,
      - radius en steellengte worden via de pinhole-formule omgerekend naar
        wat de live-camera (325 mm) zou zien (NAAR_LIVE_EENHEDEN = True).
  * Normalisatie gebeurt met vaste min/max uit deze trainset, opgeslagen in
    KNN_normalisatie.npz. Het live-programma laadt dezelfde waarden.
  * saved_cor/*.png zijn al gecorrigeerd -> hier GEEN undistort meer.
"""
import os
import re
import glob
import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ============================== INSTELLINGEN ==============================
FOTO_PATROON = 'saved_cor/*.png'

HOOGTE_CAMERA_FOTO = 500.0   # mm camera -> grond bij alle foto's
HOOGTE_CAMERA_LIVE = 325.0   # mm camera -> grond waarop bestand 1 is afgesteld
CX_LIVE = 175.47             # cx uit get_distance() in bestand 1 (px, in de crop)
S_TCP_OFFSET = 50.0          # 'S_tcp = 50 + afstand' uit hoeken_vinden()

DEFAULT_POSITIE = [180.0, 180.0]   # hoek onbekend -> niet-geroteerde crop (zoals bestand 1)
FRAME_W, FRAME_H = 1280, 720
X_LIN_SPOOF = 212

NAAR_LIVE_EENHEDEN = True    # True: radius/lijn omgerekend naar live-camera; False: mm
NORM_PERCENTIEL = (0, 100)   # bv. (1, 99) als losse uitschieters de schaal verpesten
DEBUG_CROPS = 5              # aantal crops dat naar debug_crops/ wordt geschreven (0 = uit)
BOOTSTRAP_N = 10             # trainset = mediaan van N foto's (zelfde aggregatie als live); 0 = losse foto's
BOOTSTRAP_PER_LABEL = 60     # aantal trainpunten per label bij bootstrap

# ============================== CALIBRATIE ================================
_cal = np.load("camera_calibration.npz")
mtx = _cal["camera_matrix"]
dist = _cal["dist_coeffs"]
newcameramtx = _cal["newcameramtx"]
fx = mtx[0, 0]
fy = mtx[1, 1]

_midden = cv.undistortPoints(
    np.array([[[FRAME_W / 2, FRAME_H / 2]]], dtype=np.float32),
    mtx, dist, None, newcameramtx)
x_cen_cor = int(_midden[0][0][0])


# ============ DEEL A: DETECTIE (gelijk aan bestand 1) =====================
def get_distance(xa, xb, hoogte_camera, cx):
    """Zelfde als bestand 1, maar cx is nu een parameter i.p.v. vaste 175.47."""
    basisafstand = 100
    xa = np.abs(xa - cx)
    xb = np.abs(xb - cx)

    AAf = np.sqrt(xa ** 2 + fx ** 2)
    BBf = np.sqrt(xb ** 2 + fx ** 2)

    Fi2 = np.arcsin(xb / BBf)
    Fi1 = np.pi * 0.5 - Fi2
    Theta2 = np.arcsin(xa / AAf)
    Theta1 = np.pi * 0.5 - Theta2
    Alfa = Fi2 + Theta2

    with np.errstate(divide='ignore', invalid='ignore'):
        AP = np.sin(Fi1) * (basisafstand / np.sin(Alfa))
        BP = np.sin(Theta1) * (basisafstand / np.sin(Alfa))

    Za = np.sin(Theta1) * AP
    Zb = np.sin(Fi1) * BP
    return hoogte_camera - Za, hoogte_camera - Zb


def hoeken_vinden(rotatie_graden, centerpunt, centered, afstand):
    S_tcp = S_TCP_OFFSET + afstand
    Breedte_tafel = 200

    if centerpunt == 1:
        Center, fb = 360, fy
    else:
        Center, fb = 640, fx

    theta = np.deg2rad(rotatie_graden - 180)
    Xa = -1 * (Breedte_tafel / 2) * np.cos(theta)
    Za = S_tcp - (Breedte_tafel / 2) * np.sin(theta)
    Xb = (Breedte_tafel / 2) * np.cos(theta)
    Zb = S_tcp + (Breedte_tafel / 2) * np.sin(theta)

    if centered == 1:
        return Center + (fb * Xa / Za), Center + (fb * Xb / Zb)
    return (fb * Xa / Za), (fb * Xb / Zb)


def _venster(deel, positie, afstand, x_deler):
    """Eén bounding box (links of rechts), identiek aan fotosplitter in bestand 1."""
    y_1, y_2 = hoeken_vinden(positie[0], 1, 0, afstand)
    x_1, x_2 = hoeken_vinden(positie[1], 0, 0, afstand)

    h, w = deel.shape[:2]
    cy, cx = int(h / 2), int(w / x_deler)
    y_1 += cy; y_2 += cy
    x_1 += cx; x_2 += cx

    y_s = max(0, min(min(int(y_1), int(y_2)), h))
    y_e = max(0, min(max(int(y_1), int(y_2)), h))
    x_s = max(0, min(min(int(x_1), int(x_2)), w))
    x_e = max(0, min(max(int(x_1), int(x_2)), w))
    return deel[y_s:y_e, x_s:x_e]


def fotosplitter(frame, positie, afstand):
    if frame is None or frame.size == 0:
        return None, None, None, None, None

    linker_frame = frame[0:FRAME_H, X_LIN_SPOOF:x_cen_cor]
    rechter_frame = frame[0:FRAME_H, x_cen_cor:FRAME_W]

    klein_links = _venster(linker_frame, positie, afstand, 1.7)
    klein_rechts = _venster(rechter_frame, positie, afstand, 2.3)

    if klein_links.size == 0 or klein_rechts.size == 0 \
            or klein_links.shape[0] != klein_rechts.shape[0]:
        return None, None, None, None, None

    klein_combi = np.hstack((klein_links, klein_rechts))
    return linker_frame, rechter_frame, klein_links, klein_rechts, klein_combi


def _gradient(binair):
    sx = cv.Sobel(binair, cv.CV_64F, 1, 0, ksize=1)
    sy = cv.Sobel(binair, cv.CV_64F, 0, 1, ksize=1)
    return cv.convertScaleAbs(cv.magnitude(sx, sy))


def preprocessing(img, schaal=1.0):
    """Zelfde stappen als bestand 1. Alleen het sluiten (dilate+erode, 10x live)
    schaalt mee met de pixelschaal van de crop."""
    kernel = np.ones((2, 2), np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 205, 255, 0)

    # lijn-pad (voor Find_lines)
    eroded_lijn = cv.erode(thresh, kernel, iterations=1)
    edged_lijn = cv.dilate(eroded_lijn, kernel, iterations=3)
    gradient_lijn = _gradient(edged_lijn)

    # diepte-pad
    # it = max(1, int(round(10 * schaal)))
    dilate = cv.dilate(thresh, kernel, iterations=65)
    eroded_voor_diepte = cv.erode(dilate, kernel, iterations=65)
    gradient_diepte = _gradient(eroded_voor_diepte)

    return gradient_diepte, eroded_voor_diepte, gradient_lijn


def find_lines(gradient, schaal=1.0):
    """Gemiddelde lijnlengte in pixels (0 = geen steel). Hough-parameters
    schalen mee met de pixelschaal; bij schaal=1 identiek aan bestand 1."""
    drempel = max(1, int(round(83 * schaal)))
    linesP = cv.HoughLinesP(gradient, 1, np.pi / 180, drempel, None,
                            50 * schaal, 15 * schaal)
    if linesP is None or len(linesP) <= 2:
        return 0.0

    lengtes, hoeken = [], []
    for l in linesP:
        x1, y1, x2, y2 = map(int, l.flatten())
        lengte = math.hypot(x1 - x2, y1 - y2)
        lengtes.append(lengte)
        hoeken.append(math.acos(min(1.0, abs(x1 - x2) / lengte)))

    gem_hoek = sum(hoeken) / len(hoeken)
    overeenkomend = [h for h in hoeken if abs(gem_hoek - h) < 0.07]
    if len(overeenkomend) >= 2:
        return sum(lengtes) / len(lengtes)
    return 0.0


def euclidian_afstand(eroded_voor_diepte, hoogte_camera, rand, cx):
    distance = cv.distanceTransform(eroded_voor_diepte, cv.DIST_L2, 5)
    _, r_l, _, c_l = cv.minMaxLoc(distance[:, :rand])
    _, r_r, _, c_r = cv.minMaxLoc(distance[:, rand:])
    za, _ = get_distance(c_l[0], c_r[0] + rand, hoogte_camera, cx)
    return za, r_l, r_r


def wittepunten_afstand(gradient_diepte, hoogte_camera, rand, cx):
    """Alleen de links-min/rechts-min combinatie (lengteZa_witL) wordt in
    bestand 1 gebruikt; dit is de snelle, vectoriële versie daarvan."""
    kolommen = np.flatnonzero((gradient_diepte > 0).any(axis=0))
    links = kolommen[kolommen <= rand]
    rechts = kolommen[kolommen > rand]
    if links.size == 0 or rechts.size == 0:
        return np.nan
    za, _ = get_distance(int(links.min()), int(rechts.min()), hoogte_camera, cx)
    return za


def gewogen_avg(waarden, verwacht=100.0, tolerantie=15.0):
    w = np.asarray(waarden, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.nan
    gewichten = np.exp(-((w - verwacht) ** 2) / (2 * tolerantie ** 2))
    if gewichten.sum() < 1e-12:          # beide ver van 'verwacht' -> gewoon gemiddelde
        return float(w.mean())
    return float(np.average(w, weights=gewichten))


# ============ DEEL B: FEATURES + NORMALISATIE (ook in live gebruiken) =====
def bereken_features(img_ck, hoogte_camera):
    """
    img_ck: hstack(klein_links, klein_rechts) uit fotosplitter.
    Geeft [hoogte_mm, radius_feature, lijn_feature] terug (zelfde volgorde als
    de trainset van bestand 1) en een dict met extra info.
    Live aanroepen met hoogte_camera = 325, foto's met 500.
    """
    schaal = (HOOGTE_CAMERA_LIVE + S_TCP_OFFSET) / (hoogte_camera + S_TCP_OFFSET)
    offset = img_ck.shape[1] / 4
    rand = int(offset * 2)
    cx = CX_LIVE * schaal                 # = 175.47 live, kleiner bij een kleinere crop

    gradient_diepte, eroded, gradient_lijn = preprocessing(img_ck, schaal)
    lijn_px = find_lines(gradient_lijn, schaal)
    za_euc, r_l, r_r = euclidian_afstand(eroded, hoogte_camera, rand, cx)
    za_wit = wittepunten_afstand(gradient_diepte, hoogte_camera, rand, cx)

    if r_l <= 0 or r_r <= 0:
        raise ValueError("geen object gevonden in één van de beide helften")

    hoogte = gewogen_avg([za_wit, za_euc])
    if not np.isfinite(hoogte):
        raise ValueError("afstandsberekening gaf geen geldige waarde")

    z_obj = hoogte_camera - hoogte        # afstand camera -> bovenkant object
    if z_obj <= 0:
        raise ValueError(f"onmogelijke hoogte ({hoogte:.0f} mm)")

    r_px = (r_l + r_r) / 2
    radius_mm = r_px * z_obj / fx
    lijn_mm = lijn_px * z_obj / fx

    if NAAR_LIVE_EENHEDEN:
        # Wat zou de live-camera (325 mm) zien bij een object van deze hoogte?
        z_live = max(HOOGTE_CAMERA_LIVE - hoogte, 1.0)
        r_px_live = radius_mm * fx / z_live
        radius_feat = HOOGTE_CAMERA_LIVE / r_px_live      # formule uit bestand 1
        lijn_feat = lijn_mm * fx / z_live
    else:
        radius_feat, lijn_feat = radius_mm, lijn_mm

    info = dict(offset=offset, cx=cx, schaal=schaal, r_px=r_px, lijn_px=lijn_px,
                radius_mm=radius_mm, lijn_mm=lijn_mm,
                za_euc=float(za_euc), za_wit=float(za_wit))
    return np.array([hoogte, radius_feat, lijn_feat], dtype=np.float64), info


def fit_normalisatie(raw):
    """Min/max per feature uit de trainset. Lijn: 0 = geen steel, dus min = 0."""
    lo = np.percentile(raw, NORM_PERCENTIEL[0], axis=0)
    hi = np.percentile(raw, NORM_PERCENTIEL[1], axis=0)
    lo[2] = 0.0
    hi = np.where(hi - lo < 1e-9, lo + 1.0, hi)
    return lo, hi


def normaliseer(raw, lo, hi):
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def robuuste_meting(featlijst, min_geldig=6):
    """Mediaan over meerdere frames, met uitbijterfilter.
    featlijst: lijst [hoogte, radius, lijn] uit bereken_features() (mislukte frames weglaten).
    Live: ~10 frames verzamelen en dit aanroepen; None = te onrustig, opnieuw meten."""
    F = np.array(featlijst, dtype=np.float64)
    if len(F) < min_geldig:
        return None
    med = np.median(F[:, :2], axis=0)
    mad = np.median(np.abs(F[:, :2] - med), axis=0)
    tol = np.maximum(3 * 1.4826 * mad, 0.05 * np.abs(med))    # ondergrens tegen mad ~ 0
    F = F[np.all(np.abs(F[:, :2] - med) <= tol, axis=1)]
    if len(F) < min_geldig:
        return None
    lijn = F[:, 2]
    heeft_steel = (lijn > 0).mean() > 0.5                       # meerderheid beslist
    lijn_val = float(np.median(lijn[lijn > 0])) if heeft_steel else 0.0
    return np.array([*np.median(F[:, :2], axis=0), lijn_val])


def bootstrap_trainset(raw, labels, n=10, per_label=60, seed=0):
    """Trainpunten met dezelfde aggregatie als live: mediaan van n willekeurige foto's per label."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    X, y = [], []
    for l in sorted(set(labels)):
        idx = np.flatnonzero(labels == l)
        for _ in range(per_label):
            kies = rng.choice(idx, size=min(n, len(idx)), replace=True)
            m = robuuste_meting(list(raw[kies]), min_geldig=1)
            if m is not None:
                X.append(m)
                y.append(l)
    return np.array(X), y


# ============================== HOOFDPROGRAMMA ============================
def get_label(pad):
    naam = os.path.splitext(os.path.basename(pad))[0]
    return re.sub(r'\d+$', '', naam) or naam


def main():
    paden = sorted(glob.glob(FOTO_PATROON))
    print(f"{len(paden)} foto's gevonden ({FOTO_PATROON}), "
          f"camerahoogte foto = {HOOGTE_CAMERA_FOTO:.0f} mm")

    raw, labels, bestanden, mislukt = [], [], [], []
    eerste_info = None

    for i, pad in enumerate(paden):
        frame = cv.imread(pad)
        if frame is None:
            mislukt.append((pad, "kan bestand niet lezen"))
            continue
        if frame.shape[:2] != (FRAME_H, FRAME_W):
            mislukt.append((pad, f"formaat {frame.shape[1]}x{frame.shape[0]}, verwacht "
                                 f"{FRAME_W}x{FRAME_H}"))
            continue

        # saved_cor is al gecorrigeerd: geen cv.undistort!
        _, _, _, _, img_ck = fotosplitter(frame, DEFAULT_POSITIE, HOOGTE_CAMERA_FOTO)
        if img_ck is None or img_ck.size == 0:
            mislukt.append((pad, "bounding box leeg"))
            continue

        if i < DEBUG_CROPS:
            os.makedirs("debug_crops", exist_ok=True)
            cv.imwrite(os.path.join("debug_crops", os.path.basename(pad)), img_ck)

        try:
            feat, info = bereken_features(img_ck, HOOGTE_CAMERA_FOTO)
        except ValueError as e:
            mislukt.append((pad, str(e)))
            continue

        if eerste_info is None:
            eerste_info = info
        raw.append(feat)
        labels.append(get_label(pad))
        bestanden.append(pad)

    if mislukt:
        print(f"\n{len(mislukt)} foto's overgeslagen:")
        for pad, reden in mislukt:
            print(f"  {pad}: {reden}")
    if not raw:
        print("Geen geldige data verzameld!")
        return

    if eerste_info:
        print(f"\nControle crop: offset = {eerste_info['offset']:.1f} px, "
              f"cx = {eerste_info['cx']:.1f} px, pixelschaal t.o.v. live = "
              f"{eerste_info['schaal']:.3f}")

    raw = np.array(raw)
    lo, hi = fit_normalisatie(raw)
    norm = normaliseer(raw, lo, hi)

    unieke_labels = sorted(set(labels))
    label_naar_nummer = {l: i for i, l in enumerate(unieke_labels)}
    terug = np.array([label_naar_nummer[l] for l in labels], dtype=np.int32).reshape(-1, 1)

    # ---- overzicht per label (ruwe waarden, handig om met live te vergelijken)
    print("\nPer label (ruw): hoogte [mm] | radius-feature | lijn-feature")
    for l in unieke_labels:
        m = np.array([lab == l for lab in labels])
        mu, sd = raw[m].mean(axis=0), raw[m].std(axis=0)
        print(f"  {l:<32} n={m.sum():<3} "
              f"{mu[0]:7.1f}±{sd[0]:<5.1f} {mu[1]:7.2f}±{sd[1]:<5.2f} {mu[2]:7.1f}±{sd[2]:<5.1f}")
    print(f"\nNormalisatie lo = {np.round(lo, 3)}  hi = {np.round(hi, 3)}")

    # ---- trainset: mediaan van BOOTSTRAP_N foto's (zelfde aggregatie als live) of losse foto's
    np.save("KNN_trainset_enkel.npy", norm.astype(np.float32))
    np.save("KNN_labels_enkel.npy", terug)
    if BOOTSTRAP_N > 0:
        boot_raw, train_labels = bootstrap_trainset(raw, labels, BOOTSTRAP_N, BOOTSTRAP_PER_LABEL)
        train_x = normaliseer(boot_raw, lo, hi)
    else:
        train_x, train_labels = norm, labels
    train_y = np.array([label_naar_nummer[l] for l in train_labels],
                       dtype=np.int32).reshape(-1, 1)
    np.save("KNN_trainset.npy", train_x.astype(np.float32))
    np.save("KNN_labels.npy", train_y)
    np.save("KNN_features_raw.npy", raw.astype(np.float32))
    np.savez("KNN_normalisatie.npz", lo=lo, hi=hi)
    with open("KNN_labelnamen.txt", "w") as f:
        for l, n in label_naar_nummer.items():
            f.write(f"{n}\t{l}\n")
    print(f"\nOpgeslagen: KNN_trainset.npy {train_x.shape}, KNN_labels.npy, "
          f"KNN_normalisatie.npz, KNN_labelnamen.txt, KNN_features_raw.npy")

    # ---- plot (genormaliseerde hoogte vs radius)
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(8, 6))
    for i, l in enumerate(unieke_labels):
        m = np.array([lab == l for lab in train_labels])
        plt.scatter(train_x[m, 0], train_x[m, 1], color=cmap(i % 10), label=l)
    plt.legend(title="Data Labels")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Genormaliseerde hoogte")
    plt.ylabel("Genormaliseerde radius-feature")
    plt.title("Trainset (detectie gelijk aan live-programma)")
    plt.show()


if __name__ == "__main__":
    main()