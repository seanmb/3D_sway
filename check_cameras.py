import cv2

print("Scanning cameras with DSHOW backend...")
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        fcc   = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join(chr((fcc >> (8 * j)) & 0xFF) for j in range(4))
        print(f"  index {i}: {w}x{h} @ {fps:.0f}fps  codec={codec!r}")
        cap.release()
    else:
        print(f"  index {i}: not available")
