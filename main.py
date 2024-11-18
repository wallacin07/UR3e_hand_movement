import cv2 
import mediapipe as mp
import URBasic
import time
import math

# Constants for hand tracking
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 400
SCREEN_CENTER = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

# Initialize MediaPipe Hands module
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Initialize robot settings
ROBOT_IP = '169.254.41.22'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value

# Initial joint positions (in radians)
initial_joint_positions = [-1.7075,  -2.298638288760185, -1.5654999210629865, -0.7151, 1.5962, -0.0105]
# initial_joint_positions = [-1.7075, -1.4654, -1.5655, -0.1151, 1.5962, -0.0105]

video_resolution = (SCREEN_WIDTH, SCREEN_HEIGHT)
video_midpoint = (int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2))

vs = cv2.VideoCapture(0)  # OpenCV video capture
vs.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])

# Initialize robot with URBasic
print("initializing robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialized")
time.sleep(1)

robot.movej(q=initial_joint_positions, a=ACCELERATION, v=VELOCITY)

robot_position = [0, 0, 0]  # 3D position: X, Y, Z
origin = None

robot.init_realtime_control()
time.sleep(1)

# Function to find hands using MediaPipe
def find_hands(image):
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = Hands.process(frame_rgb)
    hand_points = results.multi_hand_landmarks
    if hand_points:
        for points in hand_points:
            mpDraw.draw_landmarks(image, points, hands.HAND_CONNECTIONS)
    return hand_points, image

# Function to control robot based on hand position
# Function to control robot based on hand position
def move_to_hand(hand_landmarks, robot_pos):
    if hand_landmarks:
        # Using wrist (landmark 0) as the reference point
        wrist = hand_landmarks[0]
        wrist_position = (wrist.x, wrist.y, wrist.z)  # Normalized coordinates of the wrist

        # Print wrist position for debugging
        print(f"Wrist Position (Normalized): {wrist_position}")

        # Convert normalized wrist coordinates to screen coordinates
        screen_x = wrist_position[0] * SCREEN_WIDTH
        screen_y = wrist_position[1] * SCREEN_HEIGHT

        # Calculate displacement from screen center
        displacement_x = (screen_x - SCREEN_CENTER[0]) / SCREEN_WIDTH
        displacement_y = (screen_y - SCREEN_CENTER[1]) / SCREEN_HEIGHT
        displacement_z = wrist_position[2]  # Z remains unchanged

        # Define scale factors for more responsive control
        scale_x = 2.5  # Scale movement in X
        scale_y = 2.5  # Scale movement in Y
        scale_z = 1    # Scale movement in Z

        # Convert to real-world coordinates (relative to initial position)
        x_pos = displacement_x * scale_x
        y_pos = displacement_y * scale_y
        z_pos = displacement_z * scale_z

        # Update robot target position (relative to current position)
        robot_target_position = [robot_pos[0] + x_pos, robot_pos[1] + y_pos, robot_pos[2] + z_pos]
        robot_target_position = check_max_xyz(robot_target_position)

        # Print robot target position for debugging
        print(f"Robot Target Position: {robot_target_position}")

        # Map wrist position to joint movements
        joint_1 = initial_joint_positions[0] + x_pos  # Control joint 1 with x
        joint_2 = initial_joint_positions[1] + y_pos  # Control joint 2 with y
        joint_3 = initial_joint_positions[2] + z_pos  # Control joint 3 with z
        joint_4 = initial_joint_positions[3]          # Keep default position
        joint_5 = initial_joint_positions[4]          # Keep default position
        joint_6 = initial_joint_positions[5]          # Keep default position

        # Apply limits to ensure the joints don't go beyond their physical range
        joint_1 = limit_joint(joint_1, -2.9, 2.9)
        joint_2 = limit_joint(joint_2, -2.5, 2.5)
        joint_3 = limit_joint(joint_3, -2.0, 2.0)
        joint_4 = limit_joint(joint_4, -3.0, 3.0)
        joint_5 = limit_joint(joint_5, -3.0, 3.0)
        joint_6 = limit_joint(joint_6, -3.0, 3.0)

        # If joint_2 is less than -2.2, lower the wrist position
        if joint_2 < -2.1:
            joint_4 -= 0.6  # Lower a bit (you can adjust the value)

        # Print calculated joint positions for debugging
        print(f"Sending joints to robot: {joint_1}, {joint_2}, {joint_3}, {joint_4}, {joint_5}, {joint_6}")

        # Send the updated joint positions to the robot
        robot.movej(q=[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6], a=ACCELERATION, v=VELOCITY)


# Function to limit joint movements within the allowed range
def limit_joint(joint_value, min_val, max_val):
    return max(min(joint_value, max_val), min_val)

# Function to limit the movement of the robot in XYZ space
def check_max_xyz(robot_target_position):
    # Limit the movement range in 3D space (you can adjust these values as needed)
    max_dist = 3
    min_dist = -3
    robot_target_position[0] = max(min(robot_target_position[0], max_dist), min_dist)
    robot_target_position[1] = max(min(robot_target_position[1], max_dist), min_dist)
    robot_target_position[2] = max(min(robot_target_position[2], max_dist), min_dist)
    return robot_target_position

# Main loop for video capture and hand tracking
while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    hand_points, frame = find_hands(frame)

    # Process hand landmarks
    results = Hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        move_to_hand(hand_landmarks, robot_position)

    # Show the processed frame
    cv2.imshow("Hand Tracking", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
vs.release()
cv2.destroyAllWindows()
