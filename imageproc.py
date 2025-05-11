import cv2
import numpy as np
import os
from numpy.linalg import norm
from deepface import DeepFace

if not os.path.exists("faces"):
    os.makedirs("faces")

known_faces = {}
for file in os.listdir("faces"):
    if file.endswith(".npy"):
        name = os.path.splitext(file)[0]
        try:
            data = np.load(os.path.join("faces", file))
            known_faces[name] = data
            print(f"✅ {name} yüklendi.")
        except Exception as e:
            print(f"⚠️ {file} yüklenemedi: {e}")

cap = cv2.VideoCapture(0)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    recognized_name = "Bilinmiyor"
    try:
        representations = DeepFace.represent(frame_rgb, enforce_detection=False)
        if representations:
            embedding = representations[0]['embedding']
            max_sim = -1

            for name, saved_embedding in known_faces.items():
                sim = cosine_similarity(embedding, saved_embedding)
                if sim > max_sim and sim > 0.7:  
                    max_sim = sim
                    recognized_name = name
    except Exception as e:
        print("Yüz tanınamadı:", e)

    
    try:
        result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except Exception as e:
        emotion = "Bilinmiyor"

    label = f"{recognized_name} - {emotion}"
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, "s: Kaydet | q: Cikis | d: Tum Yuzleri Sil", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("DeepFace Yüz & Duygu Tanıma", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        name = input("👤 Kaydedilecek kullanıcının adını girin: ").strip()
        if name:
            try:
                rep = DeepFace.represent(frame_rgb, enforce_detection=True)
                embedding = rep[0]["embedding"]
                np.save(os.path.join("faces", f"{name}.npy"), embedding)
                known_faces[name] = embedding
                print(f"✅ {name} için yüz kaydı yapıldı.")

                save_frame = frame.copy()
                cv2.putText(save_frame, f"{name} KAYDEDILDI", (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Face Recognition", save_frame)
                cv2.waitKey(1000)
            except Exception as e:
                print("🚫 Yüz kaydı başarısız:", e)

    elif key == ord('d'):
        for file in os.listdir("faces"):
            os.remove(os.path.join("faces", file))
        known_faces.clear()
        print("❌ Tüm yüz kayıtları silindi.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
