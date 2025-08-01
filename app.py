from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import math
import mediapipe as mp

app = Flask(__name__)
expression = ""
last_click = 0
click_delay = 1.0
hovered = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

keys = [["7", "8", "9", "+"],
        ["4", "5", "6", "-"],
        ["1", "2", "3", "*"],
        ["C", "0", "=", "/"]]

# Minimal calculator logic
def safe_eval(expr):
    try:
        if not expr: return "0"
        if not all(c in "0123456789+-*/.()" for c in expr): return "Err"
        result = eval(expr)
        return str(int(result) if isinstance(result, float) and result.is_integer() else result)
    except:
        return "Err"

def draw_calc(frame, value):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x, y = 60+j*90, 140+i*90
            cv2.rectangle(frame, (x, y), (x+70, y+70), (70,200,255) if value==key else (50,50,50), -1)
            cv2.putText(frame, key, (x+20, y+45), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,0), 2)
    cv2.rectangle(frame, (60,40), (410,110), (30,30,30), -1)
    disp = expression if expression else "0"
    clr = (0,255,255) if disp!="Err" else (0,0,255)
    cv2.putText(frame, disp[-12:], (70,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, clr, 3)

def process_hand(frame):
    global expression, last_click, hovered
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    hovered = None
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        h,w=frame.shape[:2]
        pts = [(int(p.x*w),int(p.y*h)) for p in lm.landmark]
        thumb, index = pts[4], pts[8]
        cx,cy = (thumb[0]+index[0])//2, (thumb[1]+index[1])//2
        cv2.circle(frame, (cx,cy), 13, (0,255,255), -1)
        length = math.hypot(index[0]-thumb[0], index[1]-thumb[1])
        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                x, y = 60+j*90, 140+i*90
                if x < cx < x+70 and y < cy < y+70:
                    hovered = key
                    if length < 40 and time.time()-last_click > click_delay:
                        if key == 'C':
                            expression = ''
                        elif key == '=':
                            expression = safe_eval(expression)
                        else:
                            if expression == "Err": expression=""
                            expression += key
                        last_click = time.time()
    draw_calc(frame, hovered)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        process_hand(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manual_click', methods=['POST'])
def manual_click():
    global expression
    v = request.json.get('value')
    if v == 'C':
        expression = ''
    elif v == '=':
        expression = safe_eval(expression)
    else:
        if expression == "Err": expression=""
        expression += v
    return jsonify({"expression": expression if expression else "0"})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
