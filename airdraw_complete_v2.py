# airdraw_full.py
"""
AirDraw Ultimate - Gesture Color & Game Menu
Merged + TTS readback on save
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import pyttsx3
import speech_recognition as sr
from tensorflow.lite.python.interpreter import Interpreter as tflite
import os
import pygame  # Added for Snake game
import math    # Added for games
import random  # Added for Snake game
import sys     # Added for Snake game
import traceback

# ---------------- CONFIG ----------------
IMG_SIZE = 28
LINE_THICKNESS = 12
IDLE_SECONDS = 1.0
MOVEMENT_THRESHOLD = 4.0
LINE_HEIGHT = 60

# ---------------- MODE BUTTONS ----------------
button_w, button_h = 100, 60
button_x, button_y_start = 10, 50
button_gap = 20
current_mode = "Text"

colors_inactive = {"Text":(200,220,255), "Numeric":(220,255,220), "Save":(255,220,200)}
colors_active = {"Text":(120,180,255), "Numeric":(120,255,120), "Save":(255,150,100)}

# ---------------- LOAD TFLITE MODELS ----------------
interpreter_text = tflite(model_path="combined_letters_fallback_cnn.tflite")
interpreter_text.allocate_tensors()
input_details_text = interpreter_text.get_input_details()
output_details_text = interpreter_text.get_output_details()
classes_text = [chr(i) for i in range(65,91)] + [chr(i) for i in range(97,123)]

interpreter_numeric = tflite(model_path="digits_fallback_cnn.tflite")
interpreter_numeric.allocate_tensors()
input_details_num = interpreter_numeric.get_input_details()
output_details_num = interpreter_numeric.get_output_details()
classes_numeric = [str(i) for i in range(10)]

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils # Added for TicTacToe game
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------------- UI ----------------
DRAW_H, DRAW_W = 360, 480
TEXT_H = 180
WINDOW_NAME = "AirDraw Ultimate"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

draw_canvas = np.ones((DRAW_H, DRAW_W,3), np.uint8)*255
text_canvas = np.ones((TEXT_H,DRAW_W*2,3), np.uint8)*200
predicted_letters = []
last_pos = None
last_move_time = time.time()
soft_preview = ""

# ---------------- PALETTE & GAME ----------------
palette_open = False
gesture_start_time = None
palette_colors = [(0,0,0),(0,0,255),(0,255,0),(255,0,0)]
palette_names = ["Black","Red","Green","Blue"]
selected_color = (0,0,0)
PALETTE_HOLD_SECONDS = 2.0

game_menu_active = False
GAME_HOLD_SECONDS = 2.0
gesture_five_start_time = None

# ---------------- VOICE ----------------
engine = pyttsx3.init()
# reduce abrupt destructor issues by setting driver properties if desired
recognizer = sr.Recognizer()

# ---------------- GLOBAL CAPTURE ----------------
# Must be global to be released/re-acquired by games
cap = None 

# ---------------- HELPERS (AirDraw) ----------------
def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, r=12):
    x1,y1 = pt1; x2,y2 = pt2
    cv2.rectangle(img,(x1+r,y1),(x2-r,y2),color,thickness)
    cv2.rectangle(img,(x1,y1+r),(x2,y2-r),color,thickness)
    cv2.circle(img,(x1+r,y1+r),r,color,thickness)
    cv2.circle(img,(x2-r,y1+r),r,color,thickness)
    cv2.circle(img,(x1+r,y2-r),r,color,thickness)
    cv2.circle(img,(x2-r,y2-r),r,color,thickness)

def preprocess_canvas_for_model(canvas_img):
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    if np.count_nonzero(thresh) == 0: return None
    x,y,w,h = cv2.boundingRect(thresh)
    pad=6
    x1,y1=max(x-pad,0),max(y-pad,0)
    x2,y2=min(x+w+pad,thresh.shape[1]),min(y+h+pad,thresh.shape[0])
    roi = thresh[y1:y2,x1:x2]
    roi = cv2.resize(roi,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    roi = roi.astype(np.float32)/255.0
    return roi.reshape(1,IMG_SIZE,IMG_SIZE,1)

def predict_from_canvas(canvas_img, interpreter, input_details, output_details, classes):
    inp = preprocess_canvas_for_model(canvas_img)
    if inp is None: return None, 0.0
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(preds))
    return classes[idx], float(preds[idx])

def append_voice_text():
    global predicted_letters
    with sr.Microphone() as source:
        print("[Voice] Listening...")
        try:
            # a bit more generous phrase time limits to avoid timeouts
            audio = recognizer.listen(source, timeout=4, phrase_time_limit=6)
            text = recognizer.recognize_google(audio)
            for ch in text:
                x_jitter = np.random.randint(-1,2)
                y_jitter = 0
                scale = 1.0 + np.random.uniform(-0.05,0.05)
                predicted_letters.append((ch, selected_color, x_jitter, y_jitter, scale))
            print(f"[Voice] Recognized: {text}")
        except sr.WaitTimeoutError:
            print("[Voice] Could not recognize. listening timed out")
        except Exception as e:
            print("[Voice] Could not recognize.", e)

def speak_text(text):
    """Speak given text synchronously (blocks until read)."""
    try:
        if not text:
            return
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        # Keep program running even if TTS fails
        print(f"[TTS Error] {e}")

def save_text_canvas(speak_after_save=True):
    """
    Save the textual canvas as an image and optionally speak the collected predicted letters.
    This preserves the existing saved-image behavior and adds TTS readback.
    """
    global predicted_letters
    if not os.path.exists("saved_notes"):
        os.makedirs("saved_notes")
    canvas = np.ones((TEXT_H, DRAW_W*2, 3), np.uint8)*200
    x_offset = 20
    current_y = 50
    text_builder = []
    for letter, color, x_jitter, y_jitter, scale in predicted_letters:
        if letter == "\n":
            x_offset = 20
            current_y += LINE_HEIGHT
            text_builder.append("\n")
            continue
        cv2.putText(canvas, letter, (x_offset+x_jitter, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        (w,h),_ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x_offset += w + 2
        text_builder.append(letter)
    filename = f"saved_notes/predicted_text_{int(time.time())}.png"
    cv2.imwrite(filename, canvas)
    print(f"[Saved] {filename}")

    # Build readable text for TTS
    joined_text = "".join(text_builder).strip()
    # Normalize newlines for speech (replace newline with pause/space)
    tts_text = joined_text.replace("\n", " ")
    if speak_after_save and tts_text:
        # Speak with a short prefix so user knows it's readback
        speak_text("Saved text says: " + tts_text)

def mouse_callback(event, x, y, flags, param):
    global current_mode, selected_color, palette_open, game_menu_active, cap
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # --- Game Menu Clicks ---
        if game_menu_active:
            # Check for Snake
            if 150 <= x <= 330 and 100 <= y <= 160:
                print("[Launcher] Starting Snake...")
                game_menu_active = False
                if cap: cap.release()
                cv2.destroyWindow(WINDOW_NAME)
                
                run_snake_game() # Run the game
                
                # --- Relaunch AirDraw ---
                print("[Launcher] Relaunching AirDraw...")
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("FATAL: Cannot reopen camera!")
                    return
                return

            # Check for TicTac
            if 150 <= x <= 330 and 180 <= y <= 240:
                print("[Launcher] Starting Tic Tac Toe...")
                game_menu_active = False
                if cap: cap.release()
                cv2.destroyWindow(WINDOW_NAME)
                
                run_tictactoe_game() # Run the game
                
                # --- Relaunch AirDraw ---
                print("[Launcher] Relaunching AirDraw...")
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("FATAL: Cannot reopen camera!")
                    return
                return

        # --- Mode buttons ---
        if button_x <= x <= button_x+button_w:
            if button_y_start <= y <= button_y_start+button_h: current_mode="Text"; return
            elif button_y_start+button_h+button_gap <= y <= button_y_start+2*button_h+button_gap: current_mode="Numeric"; return
            elif button_y_start+2*(button_h+button_gap) <= y <= button_y_start+3*button_h+2*button_gap:
                # Save and speak after saving
                save_text_canvas(speak_after_save=True)
                return
        
        # --- Palette selection ---
        if palette_open:
            p_x, p_y = 200, 50; box_w, box_h = 250,100
            if p_x <= x <= p_x+box_w and p_y <= y <= p_y+box_h:
                for i, c in enumerate(palette_colors):
                    cx1 = p_x + 10 + i*55; cy1 = p_y + 35
                    cx2, cy2 = cx1+45, cy1+45
                    if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                        selected_color = c
                        palette_open = False
                        print(f"[Palette] Selected {palette_names[i]}")
                        return

# ---------------- Finger Check (AirDraw) ----------------
def check_three_fingers_up(hand_landmarks):
    tips = [8,12,16]
    pips = [6,10,14]
    extended = []
    for tip, pip in zip(tips, pips):
        extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y - 0.02)
    return all(extended)

def check_five_fingers_up(hand_landmarks):
    tips = [4,8,12,16,20]
    pips = [2,6,10,14,18]
    extended = []
    for tip, pip in zip(tips, pips):
        extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y - 0.02)
    return all(extended)


# -----------------------------------------------------------------
# -------------------- GAME 1: TIC TAC TOE ------------------------
# -----------------------------------------------------------------

def run_tictactoe_game():
    print("[Game] Starting Tic Tac Toe...")

    # --- TTT Helper Functions ---
    def distance(p1,p2):
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

    def ttt_reset_board(state):
        state['board'][:] = [['' for _ in range(3)] for _ in range(3)]
        state['game_over'] = False
        state['hold_start_time'] = None
        state['current_cell'] = None

    def place_marker(board, row, col, marker):
        if board[row][col]=='':
            board[row][col]=marker
            return True
        return False

    def check_winner(bd):
        for i in range(3):
            if bd[i][0]==bd[i][1]==bd[i][2]!='': return bd[i][0]
            if bd[0][i]==bd[1][i]==bd[2][i]!='': return bd[0][i]
        if bd[0][0]==bd[1][1]==bd[2][2]!='': return bd[0][0]
        if bd[0][2]==bd[1][1]==bd[2][0]!='': return bd[0][2]
        for row in bd:
            for cell in row:
                if cell=='': return None
        return 'Draw'

    def draw_board(img, board, highlight_cell=None, hold_progress=0):
        for i in range(1,3):
            cv2.line(img,(0,i*200),(600,i*200),(255,255,255),5)
            cv2.line(img,(i*200,0),(i*200,600),(255,255,255),5)
        if highlight_cell:
            row,col = highlight_cell
            cv2.rectangle(img,(col*200+5,row*200+5),((col+1)*200-5,(row+1)*200-5),(0,255,0),3)
            cx, cy = col*200+180, row*200+20
            cv2.circle(img, (cx,cy), 15, (0,255,0), 2)
            cv2.circle(img, (cx,cy), int(15*hold_progress), (0,255,0), -1)
        for i in range(3):
            for j in range(3):
                if board[i][j]=='X':
                    cv2.line(img,(j*200+50,i*200+50),(j*200+150,i*200+150),(0,0,255),5)
                    cv2.line(img,(j*200+150,i*200+50),(j*200+50,i*200+150),(0,0,255),5)
                elif board[i][j]=='O':
                    cv2.circle(img,(j*200+100,i*200+100),50,(255,0,0),5)

    def minimax(bd, depth, is_max, player, computer):
        winner = check_winner(bd)
        if winner==computer: return 10-depth
        elif winner==player: return depth-10
        elif winner=='Draw': return 0
        if is_max:
            best=-1000
            for i in range(3):
                for j in range(3):
                    if bd[i][j]=='':
                        bd[i][j]=computer
                        score=minimax(bd,depth+1,False, player, computer)
                        bd[i][j]=''
                        best=max(best,score)
            return best
        else:
            best=1000
            for i in range(3):
                for j in range(3):
                    if bd[i][j]=='':
                        bd[i][j]=player
                        score=minimax(bd,depth+1,True, player, computer)
                        bd[i][j]=''
                        best=min(best,score)
            return best

    def ai_move(board, player, computer):
        best_score=-1000
        move=None
        for i in range(3):
            for j in range(3):
                if board[i][j]=='':
                    board[i][j]=computer
                    score=minimax(board,0,False, player, computer)
                    board[i][j]=''
                    if score>best_score:
                        best_score=score
                        move=(i,j)
        if move: board[move[0]][move[1]]=computer

    def fingers_up(handLms):
        tips=[4,8,12,16,20]
        fingers=[]
        # Thumb
        if handLms.landmark[4].x < handLms.landmark[3].x: fingers.append(1)
        else: fingers.append(0)
        # 4 Fingers
        for id in range(1,5):
            if handLms.landmark[tips[id]].y < handLms.landmark[tips[id]-2].y: fingers.append(1)
            else: fingers.append(0)
        return fingers

    # --- TTT Setup ---
    cap_ttt = cv2.VideoCapture(0)
    if not cap_ttt.isOpened():
        print("[Game Error] Cannot open camera for TicTacToe.")
        return
        
    cap_ttt.set(3, 320)
    cap_ttt.set(4, 240)

    # Use global mp_hands, but a local hands instance
    hands_ttt = mp_hands.Hands(max_num_hands=1)

    game_state = {
        'board': [['' for _ in range(3)] for _ in range(3)],
        'player': 'X',
        'computer': 'O',
        'game_over': False,
        'hold_start_time': None,
        'current_cell': None
    }
    CLICK_THRESHOLD = 40 # Not used in hold logic, but was in original

    # --- TTT Main Loop ---
    while True:
        ret, frame = cap_ttt.read()
        if not ret:
            print("[Game Error] Failed to read frame from camera.")
            time.sleep(0.5)
            continue
            
        frame = cv2.flip(frame,1)
        small_frame = cv2.resize(frame,(200,150))
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = hands_ttt.process(rgb)

        game_screen = np.zeros((600,600,3), np.uint8)
        highlight_cell = None
        hold_progress = 0
        gx, gy = -1, -1

        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            h,w,_ = frame.shape
            x_index=int(handLms.landmark[8].x*w)
            y_index=int(handLms.landmark[8].y*h)

            gx = int(x_index*600/w)
            gy = int(y_index*600/h)
            cv2.circle(game_screen,(gx,gy),15,(0,255,0),cv2.FILLED)

            row,col=gy//200,gx//200
            if 0<=row<3 and 0<=col<3: highlight_cell=(row,col)

            # Hold to place X
            if not game_state['game_over']:
                if game_state['current_cell'] != (row,col):
                    game_state['current_cell']=(row,col)
                    game_state['hold_start_time'] = time.time()
                else:
                    if game_state['hold_start_time']:
                        hold_duration = time.time() - game_state['hold_start_time']
                        hold_progress = min(hold_duration/2.0,1.0)
                        if hold_duration >= 2.0 and place_marker(game_state['board'], row, col, game_state['player']):
                            if not check_winner(game_state['board']): 
                                ai_move(game_state['board'], game_state['player'], game_state['computer'])
                            game_state['game_over'] = bool(check_winner(game_state['board']))
                            game_state['hold_start_time'] = None
            
            # Reset gesture
            if fingers_up(handLms)==[1,1,1,1,1]: 
                ttt_reset_board(game_state)
        else:
             game_state['hold_start_time'] = None
             game_state['current_cell'] = None


        draw_board(game_screen, game_state['board'], highlight_cell, hold_progress)

        # Display winner / draw
        winner = check_winner(game_state['board'])
        if winner:
            game_state['game_over'] = True
            msg = "You Win!" if winner=='X' else ("Computer Wins!" if winner=='O' else "Draw!")
            cv2.putText(game_screen,msg,(50,300),cv2.FONT_HERSHEY_SIMPLEX,1.8,(0,255,255),4)

        # Camera feed
        game_screen[0:150,400:600] = small_frame
        cv2.imshow("Hand Gesture Tic Tac Toe", game_screen)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    # --- TTT Cleanup ---
    cap_ttt.release()
    cv2.destroyAllWindows()
    hands_ttt.close()
    print("[Game] Exiting Tic Tac Toe.")


# -----------------------------------------------------------------
# ----------------------- GAME 2: SNAKE ---------------------------
# -----------------------------------------------------------------

def run_snake_game():
    print("[Game] Starting Snake...")
    
    # --- Snake Constants ---
    FPS = 20
    GRID_SIZE = 18
    SMOOTHING = 0.18
    BODY_DELAY = 6
    MAX_PATH = 10000

    # Colors
    WHITE = (255, 255, 255)
    GREEN_HEAD = (50, 220, 50)
    GREEN_BODY = (0, 180, 0)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    # --- Snake Helper Functions ---
    def init_pygame():
        pygame.init()
        screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
        pygame.display.set_caption("Snake Tube - Hand + Arrow Control")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)
        return screen, clock, font

    def init_camera():
        cap_snake = cv2.VideoCapture(0)
        cap_snake.set(3, 320)
        cap_snake.set(4, 240)
        return cap_snake

    def get_mediapipe_hands_snake():
        # Use global mp_hands, but local hands instance
        hands_snake = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        return hands_snake

    def dist(a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def snake_touch_food_any(snake_body, food_pos):
        fx, fy = food_pos
        for seg in snake_body:
            if dist(seg, [fx, fy]) < GRID_SIZE: return True
        return False

    def get_tube_points(path, radius):
        points = []
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            dx, dy = x2 - x1, y2 - y1
            angle = math.atan2(dy, dx) + math.pi/2
            lx = x1 + radius * math.cos(angle)
            ly = y1 + radius * math.sin(angle)
            rx = x1 - radius * math.cos(angle)
            ry = y1 - radius * math.sin(angle)
            points.append(((lx, ly), (rx, ry)))
        return points

    def draw_snake_tube(surface, snake_body):
        if len(snake_body) < 2: 
            if snake_body: # Ensure not empty
                pygame.draw.circle(surface, GREEN_BODY, (int(snake_body[0][0]), int(snake_body[0][1])), GRID_SIZE)
            return

        tube_points = get_tube_points(snake_body, GRID_SIZE)
        if len(tube_points) < 2:
            for seg in snake_body:
                pygame.draw.circle(surface, GREEN_BODY, (int(seg[0]), int(seg[1])), GRID_SIZE)
            return

        left_pts = [p[0] for p in tube_points]
        right_pts = [p[1] for p in tube_points][::-1]
        polygon = left_pts + right_pts
        if len(polygon) >= 3:
            pygame.draw.polygon(surface, GREEN_BODY, polygon, 0)

        hx, hy = snake_body[0]
        pygame.draw.circle(surface, GREEN_HEAD, (int(hx), int(hy)), GRID_SIZE+3)

    # --- Snake Game Over Screen (MODIFIED) ---
    def game_over_screen(screen, font, score, WIDTH, HEIGHT):
        while True:
            screen.fill(BLACK)
            t = font.render(f"Game Over! Score: {score}", True, WHITE)
            screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2 - 60))
            r1 = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 10, 110, 50)
            r2 = pygame.Rect(WIDTH//2 + 10, HEIGHT//2 + 10, 110, 50)
            pygame.draw.rect(screen, GREEN_HEAD, r1)
            pygame.draw.rect(screen, RED, r2)
            screen.blit(font.render("Retry", True, BLACK), (r1.x+20, r1.y+10))
            screen.blit(font.render("Exit", True, BLACK), (r2.x+30, r2.y+10))
            pygame.display.update()
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False # Exit
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if r1.collidepoint(e.pos): return True # Retry
                    if r2.collidepoint(e.pos): return False # Exit
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE: return False # Exit

    def spawn_food(WIDTH, HEIGHT):
        while True:
            fx = random.randrange(GRID_SIZE, WIDTH - GRID_SIZE, GRID_SIZE)
            fy = random.randrange(GRID_SIZE, HEIGHT - GRID_SIZE, GRID_SIZE)
            if dist([fx, fy], [WIDTH//2, HEIGHT//2]) > 80: 
                return [fx, fy]

    def process_hand_input(frame, hands_snake, finger_x, finger_y, WIDTH, HEIGHT):
        try:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_snake.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                raw_x = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
                raw_y = int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
                finger_x += SMOOTHING * (raw_x - finger_x)
                finger_y += SMOOTHING * (raw_y - finger_y)
                h, w, _ = frame.shape
                cv2.circle(frame, (int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                    int(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)),
                            8, (0,0,255), -1)
                return finger_x, finger_y, frame, True
            return finger_x, finger_y, frame, False
        except Exception as e:
            print(f"[Game Error] Hand processing error: {e}")
            return finger_x, finger_y, frame, False

    # --- Snake Main Game Logic (MODIFIED for retry/exit) ---
    def main_snake():
        
        while True: # Main retry loop
            screen, clock, font = init_pygame()
            WIDTH, HEIGHT = screen.get_width(), screen.get_height()
            hands_snake = get_mediapipe_hands_snake()
            cap_snake = init_camera()
            
            if not cap_snake.isOpened():
                print("[Game Error] Cannot open camera for Snake.")
                pygame.quit()
                hands_snake.close()
                return

            path = [[WIDTH//2, HEIGHT//2]]
            snake_length = 1
            score = 0
            finger_x, finger_y = WIDTH//2, HEIGHT//2
            snake_body = [[WIDTH//2, WIDTH//2]]
            snake_body = [[WIDTH//2, HEIGHT//2]]
            food = spawn_food(WIDTH, HEIGHT)
            running = True
            paused = False
            direction = [0, 0] # dx, dy

            while running: # Single game instance loop
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        cap_snake.release(); pygame.quit(); hands_snake.close(); return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_ESCAPE:
                            cap_snake.release(); pygame.quit(); hands_snake.close(); return
                        elif ev.key == pygame.K_p:
                            paused = not paused
                        elif ev.key == pygame.K_f:
                            pygame.display.toggle_fullscreen()
                        elif ev.key == pygame.K_UP:
                            direction = [0, -GRID_SIZE/2]
                        elif ev.key == pygame.K_DOWN:
                            direction = [0, GRID_SIZE/2]
                        elif ev.key == pygame.K_LEFT:
                            direction = [-GRID_SIZE/2, 0]
                        elif ev.key == pygame.K_RIGHT:
                            direction = [GRID_SIZE/2, 0]

                if paused:
                    screen.blit(font.render("Paused", True, WHITE), (WIDTH//2 - 80, HEIGHT//2))
                    pygame.display.update()
                    clock.tick(FPS)
                    continue

                ret, frame = cap_snake.read()
                if not ret: continue

                finger_x, finger_y, frame, hand_found = process_hand_input(frame, hands_snake, finger_x, finger_y, WIDTH, HEIGHT)

                hx, hy = snake_body[0]
                if hand_found:
                    dx, dy = finger_x - hx, finger_y - hy
                    d = math.hypot(dx, dy)
                    if d > 1:
                        step = min(GRID_SIZE, d)
                        hx += step * dx / d
                        hy += step * dy / d
                else:
                    hx += direction[0]
                    hy += direction[1]

                path.insert(0, [hx, hy])
                if len(path) > MAX_PATH: path = path[:MAX_PATH]
                snake_body = path[::BODY_DELAY][:snake_length*2]

                if snake_touch_food_any(snake_body, food):
                    score += 1
                    snake_length += 1
                    food = spawn_food(WIDTH, HEIGHT)
                
                game_over = False
                # Self-collision
                for seg in snake_body[3:]:
                    if dist(snake_body[0], seg) < GRID_SIZE * 0.8:
                        game_over = True
                        break
                # Wall collision
                if hx < 0 or hx > WIDTH or hy < 0 or hy > HEIGHT:
                    game_over = True

                if game_over:
                    cap_snake.release()
                    hands_snake.close()
                    retry = game_over_screen(screen, font, score, WIDTH, HEIGHT)
                    pygame.quit() # Close the pygame window
                    if retry:
                        running = False # Break inner loop to restart outer loop
                    else:
                        return # Break outer loop and exit function
                    continue # Skip drawing this frame

                screen.fill(BLACK)
                draw_snake_tube(screen, snake_body)
                pygame.draw.rect(screen, RED, pygame.Rect(food[0]-GRID_SIZE//2, food[1]-GRID_SIZE//2, GRID_SIZE, GRID_SIZE))

                try:
                    fs = cv2.resize(frame, (340, 260))
                    fs = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
                    surf = pygame.surfarray.make_surface(np.rot90(fs))
                    screen.blit(surf, (WIDTH - 360, 20))
                except Exception as e: 
                    # print(f"CV2 blit error: {e}")
                    pass

                score_surf = font.render(f"Score: {score}", True, WHITE)
                inst_surf = font.render("ESC: Quit  P: Pause  F: Fullscreen  Arrows: Move (if no hand)", True, WHITE)
                screen.blit(score_surf, (20, 20))
                screen.blit(inst_surf, (20, 60))

                pygame.display.update()
                clock.tick(FPS)
            
            # End of inner 'while running' loop
        # End of outer 'while True' (retry) loop

    # --- Start the Snake game ---
    main_snake()
    print("[Game] Exiting Snake.")


# -----------------------------------------------------------------
# -------------------- MAIN LOOP (AirDraw) ------------------------
# -----------------------------------------------------------------

# Initialize global camera *before* setting mouse callback
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
except Exception as e:
    print(f"Failed to initialize camera: {e}")
    exit()

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
prev_time = 0

try:
    while True:
        # Safety check in case camera was not re-acquired
        if not cap.isOpened():
            print("Camera not open, attempting to re-init...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to re-init camera. Exiting.")
                break
            time.sleep(0.5)

        ret, frame = cap.read()
        if not ret: 
            # If camera temporarily fails, sleep a bit and retry
            time.sleep(0.05)
            continue
            
        frame = cv2.flip(frame, 1)
        h_cam, w_cam = frame.shape[:2]

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time>0 else 0
        prev_time = curr_time

        # select model
        if current_mode == "Text":
            interpreter = interpreter_text; in_det=input_details_text; out_det=output_details_text; classes=classes_text
        else:
            interpreter = interpreter_numeric; in_det=input_details_num; out_det=output_details_num; classes=classes_numeric

        # ---------------- Hand tracking + gestures ----------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture_three_fingers = False
        gesture_five_fingers = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fx = hand_landmarks.landmark[8].x; fy = hand_landmarks.landmark[8].y
                cur = (int(fx*DRAW_W), int(fy*DRAW_H))

                # --- Drawing only if no menu open ---
                if not palette_open and not game_menu_active:
                    if last_pos is not None:
                        dist = np.linalg.norm(np.array(cur)-np.array(last_pos))
                        if dist > MOVEMENT_THRESHOLD:
                            cv2.line(draw_canvas,last_pos,cur,selected_color,LINE_THICKNESS)
                            last_move_time = time.time()
                    else: last_move_time = time.time()
                    last_pos = cur
                    cv2.circle(frame, (int(fx*w_cam), int(fy*h_cam)), 6, (0,255,0), -1)
                else:
                    last_pos = None

                # gestures
                if check_three_fingers_up(hand_landmarks): gesture_three_fingers=True
                if check_five_fingers_up(hand_landmarks): gesture_five_fingers=True
        else:
            last_pos=None

        # --- Game Menu open logic ---
        if gesture_five_fingers:
            if gesture_five_start_time is None: gesture_five_start_time=time.time()
            elif time.time()-gesture_five_start_time >= GAME_HOLD_SECONDS and not game_menu_active:
                game_menu_active=True
                palette_open = False # Close palette if open
                print("[Game Menu] Opened!")
        else: gesture_five_start_time=None

        # --- Palette logic ---
        if gesture_three_fingers and not game_menu_active:
            if gesture_start_time is None: gesture_start_time=time.time()
            elif time.time()-gesture_start_time >= PALETTE_HOLD_SECONDS and not palette_open:
                palette_open=True
        else: gesture_start_time=None

        # ---------------- Auto Prediction ----------------
        inp = preprocess_canvas_for_model(draw_canvas)
        soft_preview=""
        if inp is not None:
            letter, conf = predict_from_canvas(draw_canvas, interpreter, 
                                                input_details_text if current_mode=="Text" else input_details_num,
                                                output_details_text if current_mode=="Text" else output_details_num,
                                                classes_text if current_mode=="Text" else classes_numeric)
            soft_preview=f"{letter} ({conf*100:.0f}%)"

        if time.time()-last_move_time > IDLE_SECONDS and inp is not None and not palette_open and not game_menu_active:
            gray=cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
            if np.count_nonzero(gray<255)>120:
                letter, conf = predict_from_canvas(draw_canvas, interpreter, 
                                                    input_details_text if current_mode=="Text" else input_details_num,
                                                    output_details_text if current_mode=="Text" else output_details_num,
                                                    classes_text if current_mode=="Text" else classes_numeric)
                x_jitter = np.random.randint(-1,2); y_jitter=0; scale=1.0 + np.random.uniform(-0.05,0.05)
                predicted_letters.append((letter, selected_color, x_jitter, y_jitter, scale))
                print(f"[Auto] Predicted: {letter} ({conf*100:.2f}%)")
                draw_canvas[:] = 255; last_pos=None; last_move_time=time.time()
                soft_preview=""

        # ---------------- UI Drawing ----------------
        draw_canvas_display = draw_canvas.copy()
        
        # --- Game Menu UI (drawn first, under buttons) ---
        if game_menu_active:
            # Snake Button
            cv2.rectangle(draw_canvas_display,(150,100),(330,160),(180,180,255),-1)
            cv2.rectangle(draw_canvas_display,(150,100),(330,160),(100,100,150),2)
            cv2.putText(draw_canvas_display,"Snake",(170,140),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
            # TicTac Button
            cv2.rectangle(draw_canvas_display,(150,180),(330,240),(180,255,180),-1)
            cv2.rectangle(draw_canvas_display,(150,180),(330,240),(100,150,100),2)
            cv2.putText(draw_canvas_display,"TicTac",(165,220),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

        # --- Mode Buttons ---
        for i,name in enumerate(["Text","Numeric","Save"]):
            y1 = button_y_start+i*(button_h+button_gap); y2=y1+button_h
            color = colors_active[name] if current_mode==name else colors_inactive[name]
            draw_rounded_rect(draw_canvas_display,(button_x,y1),(button_x+button_w,y2),color)
            cv2.putText(draw_canvas_display,name,(button_x+10,y1+40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,50,50),2,cv2.LINE_AA)

        sep_x = button_x+button_w+10
        cv2.line(draw_canvas_display,(sep_x,0),(sep_x,DRAW_H),(0,0,0),2)

        # --- Camera View ---
        cam_width = int(DRAW_W*0.8)
        cam_view = cv2.resize(frame,(cam_width,DRAW_H))
        cam_border = cv2.copyMakeBorder(cam_view,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,0))
        cam_border = cv2.resize(cam_border,(cam_border.shape[1],draw_canvas_display.shape[0]))
        cv2.putText(cam_border,f'FPS: {int(fps)}',(cam_border.shape[1]-100,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)

        main_display = np.hstack((draw_canvas_display, cam_border))

        # ---------------- Palette UI ----------------
        if palette_open:
            p_x,p_y = 200,50; p_w,p_h=250,100
            palette_box=np.ones((p_h,p_w,3),np.uint8)*230
            cv2.putText(palette_box,"Select Color",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1,cv2.LINE_AA)
            for i,c in enumerate(palette_colors):
                x1 = 10+i*55; y1=35; x2=x1+45; y2=y1+45
                cv2.rectangle(palette_box,(x1,y1),(x2,y2),c,-1)
                cv2.rectangle(palette_box,(x1,y1),(x2,y2),(100,100,100),1)
            
            # Blend palette box onto main display
            try:
                roi = main_display[p_y:p_y+p_h, p_x:p_x+p_w]
                blended = cv2.addWeighted(roi, 0.55, palette_box, 0.45, 0)
                main_display[p_y:p_y+p_h, p_x:p_x+p_w] = blended
            except Exception as e:
                print(f"Palette draw error: {e}") # Handle ROI mismatch if resizing
                pass


        # ---------------- Prediction Canvas ----------------
        text_canvas[:]=200
        x_offset=20; current_y=50
        for letter,color,x_jitter,y_jitter,scale in predicted_letters:
            if letter == "\n":
                x_offset = 20; current_y += LINE_HEIGHT
                continue
            cv2.putText(text_canvas,letter,(x_offset+x_jitter,current_y),
                        cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
            (w,h),_ = cv2.getTextSize(letter,cv2.FONT_HERSHEY_SIMPLEX,1.0,2)
            x_offset += w + 2

        if soft_preview:
            cv2.putText(text_canvas,soft_preview,(DRAW_W*2-200,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,200),2,cv2.LINE_AA)

        text_canvas_resized = cv2.resize(text_canvas,(main_display.shape[1],TEXT_H))
        final_view = np.vstack((main_display,text_canvas_resized))
        cv2.imshow(WINDOW_NAME,final_view)

        # ---------------- Keys ----------------
        key = cv2.waitKey(1)&0xFF
        if key in [27,ord('q')]: break
        elif key==ord('c'): draw_canvas[:]=255; last_pos=None
        elif key==ord('r'): draw_canvas[:]=255; predicted_letters=[]; last_pos=None
        elif key==32:
            x_jitter=np.random.randint(-1,2); y_jitter=0; scale=1.0+np.random.uniform(-0.05,0.05)
            predicted_letters.append((" ",selected_color,x_jitter,y_jitter,scale))
        elif key in [8,127]:
            if predicted_letters: predicted_letters.pop()
        elif key==13: predicted_letters.append(("\n",selected_color,0,0,1.0))
        elif key==ord('v'): append_voice_text()
        elif key==ord('p'): save_text_canvas(speak_after_save=True)
        elif key==ord('1'): selected_color=(0,0,0)
        elif key==ord('2'): selected_color=(0,0,255)
        elif key==ord('3'): selected_color=(0,255,0)
        elif key==ord('4'): selected_color=(255,0,0)
        
        # Close menus with ESC
        if key == 27:
            if palette_open: palette_open = False
            if game_menu_active: game_menu_active = False

finally:
    try:
        if cap: cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    try:
        hands.close()
    except Exception:
        pass
    # cleanly stop pyttsx3 engine to reduce destructor issues
    try:
        engine.stop()
    except Exception:
        pass
    print("Exiting AirDraw Ultimate.")
