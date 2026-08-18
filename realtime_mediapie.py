import os
import cv2
import torch

from torchvision import transforms

from model import CatDogCNN


# =====================================================
# 1. DEVICE
# =====================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("Device:", device)


# =====================================================
# 2. CLASSES
# =====================================================

classes = [
    "cat",
    "dog"
]

NUM_CLASSES = len(classes)


# =====================================================
# 3. PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/best_cat_dog_model.pth"
)


# =====================================================
# 4. CHECK MODEL
# =====================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")

    print(MODEL_PATH)

    exit()


# =====================================================
# 5. TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# =====================================================
# 6. LOAD RESNET
# =====================================================

model = CatDogCNN(
    num_classes=NUM_CLASSES
)


model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=device,

        weights_only=True

    )

)


model = model.to(device)

model.eval()


print("ResNet model loaded.")


# =====================================================
# 7. WEBCAM
# =====================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print("Webcam started.")
print("Press Q to quit.")


# =====================================================
# 8. BACKGROUND SUBTRACTOR
# =====================================================

background = cv2.createBackgroundSubtractorMOG2(

    history=500,

    varThreshold=50,

    detectShadows=False

)


# =====================================================
# 9. REAL-TIME LOOP
# =====================================================

while True:


    # =================================================
    # CAPTURE
    # =================================================

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Could not read frame.")

        break


    # =================================================
    # MIRROR
    # =================================================

    frame = cv2.flip(
        frame,
        1
    )


    # =================================================
    # FRAME SIZE
    # =================================================

    height, width, _ = frame.shape


    # =================================================
    # CREATE MASK
    # =================================================

    mask = background.apply(frame)


    # =================================================
    # REMOVE NOISE
    # =================================================

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5, 5)

    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel

    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    # =================================================
    # FIND CONTOURS
    # =================================================

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    # =================================================
    # FIND LARGEST OBJECT
    # =================================================

    if contours:

        largest_contour = max(

            contours,

            key=cv2.contourArea

        )


        area = cv2.contourArea(
            largest_contour
        )


        # Ignore small objects

        if area > 5000:


            # =============================================
            # BOUNDING BOX
            # =============================================

            x, y, w, h = cv2.boundingRect(

                largest_contour

            )


            # =============================================
            # PADDING
            # =============================================

            padding = 30


            x1 = max(
                0,
                x - padding
            )


            y1 = max(
                0,
                y - padding
            )


            x2 = min(
                width,
                x + w + padding
            )


            y2 = min(
                height,
                y + h + padding
            )


            # =============================================
            # DRAW BOX
            # =============================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            # =============================================
            # CROP
            # =============================================

            hand_crop = frame[

                y1:y2,

                x1:x2

            ]


            if hand_crop.size != 0:


                # =============================================
                # RGB
                # =============================================

                hand_rgb = cv2.cvtColor(

                    hand_crop,

                    cv2.COLOR_BGR2RGB

                )


                # =============================================
                # TRANSFORM
                # =============================================

                image_tensor = transform(

                    hand_rgb

                )


                # =============================================
                # BATCH
                # =============================================

                image_tensor = (

                    image_tensor.unsqueeze(0)

                )


                # =============================================
                # DEVICE
                # =============================================

                image_tensor = (

                    image_tensor.to(device)

                )


                # =============================================
                # PREDICTION
                # =============================================

                with torch.no_grad():

                    logits = model(

                        image_tensor

                    )


                    probabilities = (

                        torch.softmax(

                            logits,

                            dim=1

                        )

                    )


                    prediction = (

                        torch.argmax(

                            probabilities,

                            dim=1

                        )

                    )


                # =============================================
                # CLASS
                # =============================================

                predicted_class = classes[

                    prediction.item()

                ]


                # =============================================
                # CONFIDENCE
                # =============================================

                confidence = probabilities[

                    0,

                    prediction.item()

                ].item()


                # =============================================
                # DISPLAY
                # =============================================

                text = (

                    f"{predicted_class}: "

                    f"{confidence * 100:.1f}%"

                )


                cv2.putText(

                    frame,

                    text,

                    (30, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.2,

                    (0, 255, 0),

                    3

                )


    else:

        cv2.putText(

            frame,

            "Show animal",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 0, 255),

            2

        )


    # =================================================
    # SHOW
    # =================================================

    cv2.imshow(

        "Cat Dog recognization",

        frame

    )


    # =================================================
    # QUIT
    # =================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =====================================================
# 10. CLEANUP
# =====================================================

cap.release()

cv2.destroyAllWindows()


print("Program ended.")