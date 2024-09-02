import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import time
from PIL import Image
import warnings

#background purpose
page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-color:#E5B8F4;
    padding:none;
    margin:none;
    background-image: linear-gradient(0deg,#E5B8F4, #B083D7 ,#810CA8);
    }
    </style>
    """
st.set_page_config(layout="wide")
st.markdown(page_bg_img, unsafe_allow_html=True)
page = st.sidebar.selectbox("Explore Your Creative Journey", ["Home","About Epitome", "AR Transparent Board"])
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Epitome</h1>", unsafe_allow_html=True)

    carousel_images = [
        "images.jpg",  # Replace with your local image paths or URLs
        "images.jpg",
        "images.jpg"
    ]

    # Display the carousel-like effect (Manual switching for simplicity)
    carousel_placeholder = st.empty()
    carousel_placeholder.image(carousel_images[0], use_column_width=True)

    # Create the buttons to manually switch images (Optional)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            carousel_placeholder.image(carousel_images[0], use_column_width=True)
    with col2:
        if st.button("Next"):
            carousel_placeholder.image(carousel_images[1], use_column_width=True)
    # Instructions Section
    st.markdown("<br><br><br><h2 style='text-align: center; color: black;'>Instructions</h2><br><br><br>", unsafe_allow_html=True)
    instruction_images = [
    "images.jpg",  # Replace with your local image paths
    "images.jpg",
    "images.jpg",
    "images.jpg",
    "images.jpg"
    ]
    D={1:"Step 1: Launch the Application", 2:"Step 2: Choose your drawing color", 3:"Step 3: Draw on the board", 4:"Step 4: Clear the board", 5:"Step 5: Close the Application"}
    for row in range(3):  # 3 rows
        cols = st.columns(2)  # 2 columns per row
        for i in range(2):
            index = row * 2 + i
            if index < len(instruction_images):
                cols[i].image(instruction_images[index], width=350)
                cols[i].markdown(f"{D[index + 1]}<br><br><br>", unsafe_allow_html=True)

# About Epitome Page
elif page == "About Epitome":
    st.title("About Epitome: AR Transparent Drawing Board")

    # About section content
    st.markdown("""
    **Epitome** is an innovative web application designed to redefine your drawing experience through the power of augmented reality (AR). 
    Utilizing hand gestures and your device’s camera, Epitome allows users to draw, manipulate colors, and save their creations in real time, 
    offering an immersive and interactive platform built on a robust technology stack.
    """)
    
    st.header("Key Features")
    
    st.write("""
    - **Augmented Reality Drawing**  
      Epitome empowers users to draw directly in the air using hand gestures. The app captures hand movements through your webcam and translates them into lines and shapes on the screen, providing a unique and engaging drawing experience.
    
    - **Customizable Drawing Tools**  
      Tailor your creative output with a range of drawing tools. Choose from various colors and adjust the line thickness to suit your artistic needs. An easy-to-access toolbar allows for quick changes and instant clearing of the canvas.
    
    - **Real-Time Interaction**  
      Built on the Streamlit framework, Epitome ensures smooth, real-time interactions. From drawing and erasing to saving your work, all actions happen instantly, allowing for a fluid creative process.
    
    - **Elegant UI**  
      The application’s user interface is designed for modern aesthetics. A polished gradient background and carousel image display enhance the visual appeal, while the intuitive layout ensures that users can easily navigate and interact with the application.
    
    - **Performance and Responsiveness**  
      Optimized for performance, Epitome minimizes lag and provides a responsive drawing experience. An FPS counter keeps track of the application’s performance, ensuring that your interactions remain seamless.

    """)

    st.header("How It Works")
    
    st.write("""
    - **Gesture Recognition**  
      The core functionality of Epitome revolves around gesture recognition powered by the Mediapipe library. By tracking the movements of your index finger, the application allows you to draw in mid-air, translating these gestures into lines and shapes on the virtual canvas.
    
    - **Color Selection and Tools**  
      A simple and intuitive UI overlay enables users to select colors from a palette. You can easily switch colors with a gesture or clear the entire canvas with a single action, streamlining your creative workflow.
    
    - **Save and Export**  
      Once you’ve completed your masterpiece, Epitome allows you to save your drawing as an image file. With just a click, your creation can be exported and shared, preserving your digital artwork.
    """)

        
elif page == "AR Transparent Board":
    st.title("AR Transparent Board")
    st.write("This section will contain the AR Transparent Drawing Board implementation...")
    # The AR board implementation code will be added here...
    
    frame_placeholder = st.empty()
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    color_names = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN"]
    colorIndex = 0

    points = [[] for _ in range(len(colors))]

    def create_ui(width, height):
        ui_height = height // 8
        ui = np.zeros((ui_height, width, 3), dtype=np.uint8)
        
        for y in range(ui_height):
            color = [int(240 * (1 - y/ui_height))] * 3
            cv2.line(ui, (0, y), (width, y), color, 1)
        
        button_width = min(50, width // (len(colors) + 2))
        for i, color in enumerate(colors):
            x = 10 + i * (button_width + 10)
            cv2.circle(ui, (x + button_width // 2, ui_height // 2), button_width // 2 - 5, color, -1)
            cv2.circle(ui, (x + button_width // 2, ui_height // 2), button_width // 2 - 5, (0, 0, 0), 2)
            cv2.putText(ui, color_names[i][:1], (x + button_width // 2 - 5, ui_height // 2 + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.rectangle(ui, (width - 100, 10), (width - 10, ui_height - 10), (200, 200, 200), -1)
        cv2.rectangle(ui, (width - 100, 10), (width - 10, ui_height - 10), (0, 0, 0), 2)
        cv2.putText(ui, "CLEAR", (width - 90, ui_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return ui

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    frame_width = 640  # Reduced resolution
    frame_height = 480  # Reduced resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    ui = create_ui(frame_width, frame_height)
    ui_height = ui.shape[0]
    canvas = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

    def get_index_finger_tip(hand_landmarks):
        return (int(hand_landmarks.landmark[8].x * frame_width),
                int(hand_landmarks.landmark[8].y * frame_height))

    def is_index_finger_raised(hand_landmarks):
        return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

    prev_point = None
    min_distance = 5
    is_drawing = False
    line_thickness = 2

    running = True
    prev_time = time.time()
    col1, col2 = st.columns(2)
    with col1:
        start = st.button("Let's get started")
    with col2:
        stop = st.button("Stop")
    st.image("img1.jpg", caption="AR Transparent Board Illustration", use_column_width=True)
    if start:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger_tip = get_index_finger_tip(hand_landmarks)
                    
                    if is_index_finger_raised(hand_landmarks):
                        if index_finger_tip[1] <= ui_height:  # Toolbar area
                            if index_finger_tip[0] >= frame_width - 100:  # Clear button
                                for p in points:
                                    p.clear()
                                canvas.fill(255)
                                prev_point = None
                                is_drawing = False
                            else:
                                for i, x in enumerate(range(10, 10 + len(colors) * 60, 60)):
                                    if x <= index_finger_tip[0] <= x + 50:
                                        colorIndex = i
                                        break
                        else:
                            if not is_drawing:
                                prev_point = index_finger_tip
                                is_drawing = True
                            
                            if prev_point and np.linalg.norm(np.array(index_finger_tip) - np.array(prev_point)) > min_distance:
                                cv2.line(canvas, prev_point, index_finger_tip, colors[colorIndex], line_thickness)
                                prev_point = index_finger_tip
                    else:
                        prev_point = None
                        is_drawing = False

                    cv2.circle(frame, index_finger_tip, 5, colors[colorIndex], -1)

            output = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
            output[:ui_height, :] = ui

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(output, f"FPS: {int(fps)}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # cv2.imshow("AirSketch", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('s'):
                cv2.imwrite("Air_Sketch_drawing.png", canvas)
                print("Drawing saved as 'Air_Sketch_drawing.png'")
            elif key == ord('+'):
                line_thickness = min(line_thickness + 1, 10)
            elif key == ord('-'):
                line_thickness = max(line_thickness - 1, 1)
            frame_placeholder.image(output, channels="BGR")
            if stop:
                break
        cap.release()
        cv2.destroyAllWindows()