import cv2
import mediapipe as mp
import URBasic
import time
import math3d as m3d
import math

# Constants for hand tracking
MAX_DIST_Z = 0.24
MIN_DIST_Z = 0.24

# Initialize MediaPipe Hands module
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Initialize robot settings
ROBOT_IP = '169.254.41.22'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value

# Initial joint positions (in radians)
initial_joint_positions = [-1.7075, -1.4654, -1.5655, -0.1151, 1.5962, -0.0105]

video_resolution = (700, 400)
video_midpoint = (int(video_resolution[0] / 2), int(video_resolution[1] / 2))

vs = cv2.VideoCapture(0)  # OpenCV video capture
vs.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])

# Initialize robot with URBasic
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)

robot.reset_error()
print("robot initialised")
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
def move_to_hand(hand_landmarks, robot_pos):
    if hand_landmarks:
        # Let's assume we're using the wrist (landmark 0) as the main reference point
        wrist = hand_landmarks[0]
        wrist_position = (wrist.x, wrist.y, wrist.z)  # Normalized coordinates of the wrist

        # Print wrist position for debugging
        print(f"Wrist Position (Normalized): {wrist_position}")

        # Define scale factors for more responsive control in X and Y
        scale_x = 2.5  # Increase movement in X significantly
        scale_y = 2.5  # Increase movement in Y significantly
        scale_z = 1  # Keep a moderate movement in Z

        # Convert normalized coordinates to real-world coordinates
        x_pos = wrist_position[0] * scale_x
        y_pos = wrist_position[1] * scale_y
        z_pos = wrist_position[2] * scale_z

        # Update robot position with new coordinates
        robot_target_position = [robot_pos[0] + x_pos, robot_pos[1] + y_pos, robot_pos[2] + z_pos]
        robot_target_position = check_max_xy(robot_target_position)

        # Print robot target position for debugging
        print(f"Robot Target Position: {robot_target_position}")

        # Map wrist position to joint movements
        joint_1 = initial_joint_positions[0] + x_pos  # Control joint 1 with x
        joint_2 = initial_joint_positions[1] + y_pos  # Control joint 2 with y
        joint_3 = initial_joint_positions[2]  # Keep default position
        joint_4 = initial_joint_positions[3]  # Keep default position
        joint_5 = initial_joint_positions[4]  # Keep default position
        joint_6 = initial_joint_positions[5]  # Keep default position

        # Print calculated joint positions for debugging
        print(f"Sending joints to robot: {joint_1}, {joint_2}, {joint_3}, {joint_4}, {joint_5}, {joint_6}")

        # Send the updated joint positions to the robot
        robot.movej(q=[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6], a=ACCELERATION, v=VELOCITY)

# Function to limit the movement of the robot
def check_max_xy(robot_target_xy):
    max_dist = 3  # Increase maximum distance for X and Y
    robot_target_xy[0] = max(-max_dist, min(robot_target_xy[0], max_dist))
    robot_target_xy[1] = max(-max_dist, min(robot_target_xy[1], max_dist))
    return robot_target_xy

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
