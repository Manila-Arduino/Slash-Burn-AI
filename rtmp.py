# === View DJI Mini 3 Pro RTMP in OpenCV ===
# 1) Run a local RTMP server (MediaMTX) on your PC.
# 2) In DJI Fly, set RTMP Address to: rtmp://<PC_IP>/mystream
# 3) Use this Python viewer.

import os, cv2

# optional: small buffers / quicker reconnects
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtmp_buffer;0|timeout;5000000"

URL = "rtmp://192.168.1.16/live/key"  # replace with your PC's IPv4
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    # Fallback: if you restream via RTSP (MediaMTX exposes both)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000|max_delay;0"
    )
    URL = "rtsp://192.168.1.16:8554/live/key"
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

assert cap.isOpened(), "Cannot open stream"

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("DJI RTMP", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
