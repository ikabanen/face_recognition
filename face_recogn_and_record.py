import cv2
import face_recognition
import numpy as np

# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_Faces.avi', fourcc, 20.0, (640, 480))
frame_number = 0

sanders_image = face_recognition.load_image_file("Sanders.jpg")
sanders_face_encoding = face_recognition.face_encodings(sanders_image)[0]

turing_image = face_recognition.load_image_file("Turing.jpg")
turing_face_encoding = face_recognition.face_encodings(turing_image)[0]

thatcher_image = face_recognition.load_image_file("Thatcher.jpg")
thatcher_face_encoding = face_recognition.face_encodings(thatcher_image)[0]

einstein_image = face_recognition.load_image_file("Einstein.jpg")
einstein_face_encoding = face_recognition.face_encodings(einstein_image)[0]

twain_image = face_recognition.load_image_file("Twain.jpg")
twain_face_encoding = face_recognition.face_encodings(twain_image)[0]

known_face_encodings = [
    sanders_face_encoding,
    turing_face_encoding,
    thatcher_face_encoding,
    einstein_face_encoding,
    twain_face_encoding
]


known_face_names = [
    "Sanders",
    "Turing",
    "Thatcher",
    "Einstein",
    "Twain"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    frame_number += 1

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    if not ret:
        print("Can't receive video")
        break

    if process_this_frame:
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (1, 190, 200), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    out.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
