from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads model on first run

source = r"Asian market.jpg"  # for image file

#Remove the comment(#) from the lines below to test with a video file and webcam instead of an image

#source = r"Cars Moving On Road Footage.mp4"  # for video file

#source = 0  # for webcam

if isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
    model.predict(source=source, conf=0.35, show=True, save=True)
else:
    model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True)

print("Done. Check runs/ folder.")

