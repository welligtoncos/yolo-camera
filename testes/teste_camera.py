import cv2

# Substitua pela senha correta se necessário
ip = "192.168.0.108"
user = "admin"
senha = "FCBPJA46"

urls = [
    "rtsp://admin:FCIP8J46@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0",  # HD
]

for url in urls:
    print(f"🔍 Testando: {url}")
    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        print(f"✅ Sucesso com: {url}")
        break
    else:
        print("❌ Falhou")
    cap.release()
