import cv2
import mediapipe as mp

# Initialisation de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Ouvrir la caméra (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

# Vérifier si la caméra est disponible
if not cap.isOpened():
    print("❌ Impossible d’ouvrir la caméra.")
    exit()

# Configuration du modèle MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,      # Détection continue (vidéo)
    max_num_hands=2,              # Nombre maximum de mains détectées
    min_detection_confidence=0.7, # Confiance minimale pour la détection
    min_tracking_confidence=0.7   # Confiance minimale pour le suivi
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Erreur de lecture vidéo.")
            break

        # Conversion BGR → RGB (MediaPipe utilise RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des mains
        results = hands.process(frame_rgb)

        # Repassage en BGR pour l’affichage OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Dessiner les mains détectées
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dessin des points et connexions
                mp_drawing.draw_landmarks(
                    frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Afficher les coordonnées (x, y) d’un point précis
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame_bgr.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(f"Main détectée | Point {id} : x={cx}, y={cy}")

        # Afficher la vidéo
        cv2.imshow("Détection de la main - MediaPipe", frame_bgr)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer la caméra et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
