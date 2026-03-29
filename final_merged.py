from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from pupil_apriltags import Detector
import base64

app = Flask(__name__)

# -------------- AprilTag detector (tag36h11) --------------
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# -------------- Camera calibration --------------
fx = 1136.4142048524834
fy = 1136.0342023062778
cx = 541.3507176098198
cy = 539.0048399042424

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Distortion coefficients (5 params: k1, k2, p1, p2, k3)
distCoeffs = np.array(
    [2.94147566e-01, -1.93749807e+00,
     -6.31515772e-04, -3.48001419e-04,
     4.39464837e+00],
    dtype=np.float64
)

camera_params = (fx, fy, cx, cy)
tag_size = 0.14  # meters

# -------------- Room / world setup --------------
R_TAG_TO_ROOM = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

TAG_ROOM = {
    0: {"pos": np.array([3.17, 0.62, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    1: {"pos": np.array([2.44, 0.68, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    2: {"pos": np.array([2.48, 1.43, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    3: {"pos": np.array([2.59, 2.32, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    4: {"pos": np.array([2.99, 3.06, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    5: {"pos": np.array([3.25, 3.70, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    6: {"pos": np.array([1.95, 2.48, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    7: {"pos": np.array([1.23, 2.46, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    8: {"pos": np.array([0.57, 2.45, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    9: {"pos": np.array([3.10, 1.35, 2.85], dtype=np.float64),
        "R": R_TAG_TO_ROOM},

    10: {"pos": np.array([3.40, 4.34, 2.85], dtype=np.float64),
         "R": R_TAG_TO_ROOM},
}

# -------------- Temporal buffer in TAG frame --------------
ROOM_BUFFER_SIZE = 3     # how many recent samples to keep (TAG frame)
tag_positions = []        # list of np.array([x, y, z]) in TAG frame
last_tag_id = None

def median_position(positions):
    """Return per-axis median of list of 3D points (Nx3)."""
    if not positions:
        return None
    arr = np.stack(positions, axis=0)  # (N,3)
    return np.median(arr, axis=0)

def camera_pos_in_room(tag_id, cam_in_tag_vec):
    """
    cam_in_tag_vec: np.array shape (3,) camera position in TAG frame.
    Returns np.array shape (3,) in ROOM frame, or None if tag not in TAG_ROOM.
    """
    info = TAG_ROOM.get(tag_id)
    if info is None:
        return None
    p_tag_room = info["pos"]              # (3,)
    R_room_tag = info.get("R", np.eye(3)) # (3,3)
    return p_tag_room + R_room_tag @ cam_in_tag_vec

# -------------- 🔦 Brightness features + Logistic Regression (ML flashlight) --------------

def compute_brightness_features(gray_img: np.ndarray):
    """
    gray_img: 2D numpy uint8 (0..255), already resized to 320x320.
    Returns dict with meanBrightness, contrast, brightnessRange.
    """
    img = gray_img.astype(np.float32)
    mean_b = float(img.mean())
    std_b = float(img.std())
    min_b = float(img.min())
    max_b = float(img.max())
    brightness_range = max_b - min_b

    return {
        "meanBrightness": mean_b,
        "contrast": std_b,
        "brightnessRange": brightness_range,
    }

# Trained logistic regression parameters (3 features)
# Weights: [8.40797812 2.23170507 5.62473117]
# Bias: -6.2395383774836
W_LIGHT = np.array([8.40797812, 2.23170507, 5.62473117], dtype=np.float64)
B_LIGHT = -6.2395383774836

def light_prob_good(meanBrightness, contrast, brightnessRange):
    """
    Apply trained logistic regression:
    x1 = meanBrightness / 255
    x2 = contrast       / 128
    x3 = brightnessRange / 255

    p_good = sigmoid(W·x + B)
    """
    f1 = meanBrightness / 255.0
    f2 = contrast / 128.0
    f3 = brightnessRange / 255.0

    x = np.array([f1, f2, f3], dtype=np.float64)
    z = float(W_LIGHT @ x + B_LIGHT)
    p = 1.0 / (1.0 + np.exp(-z))
    return p

# ---------------- HTML (navigator + torch ML + BLE) ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Indoor Path Navigator — AprilTag + A* + Torch ML + BLE</title>
<style>
  body { font-family: Arial, sans-serif; text-align: center; background: #eef2f5; margin:0; padding:16px; }
  .controls { margin-bottom: 8px; }
  button { margin: 5px; padding: 8px 14px; font-size: 14px; cursor: pointer; border-radius: 6px; }
  #canvasWrap { display:inline-block; background:#fff; border:2px solid #333; }
  canvas { display:block; touch-action:none; max-width: 100%; height: auto; }
  #info { background: #fff; border: 1px solid #ccc; padding: 8px; margin: 10px auto; width: 94%; max-width: 760px; text-align: left; }
  #permBtn { background: #007bff; color: white; border: none; }
  #pose { background:#fff; border:1px solid #ccc; padding:8px; margin:10px auto; width:94%; max-width:760px; text-align:left; white-space:pre-wrap; }
  #lightInfo { background:#fff; border:1px solid #ccc; padding:8px; margin:10px auto; width:94%; max-width:760px; text-align:left; }
  #bleStatus { background:#fff; border:1px solid #ccc; padding:8px; margin:10px auto; width:94%; max-width:760px; text-align:left; }
</style>
</head>
<body>
<h2>🏠 Indoor Path Navigator (AprilTag start + A* guidance + Torch ML + ESP32 BLE)</h2>

<div class="controls">
  <button id="setGoal">Set Goal (click/touch map)</button>
  <button id="voiceBtn">🎙️ Voice: say object</button>
  <button id="bleConnectBtn">🔗 Connect ESP32</button>
  <button id="permBtn" style="display:none;">Enable Compass</button>
</div>

<div id="canvasWrap">
  <canvas id="navCanvas"></canvas>
</div>

<div id="info">Status: Initializing...</div>
<pre id="pose">Camera pose: waiting for tag...</pre>

<div id="lightInfo">
  <b>Light status (per frame):</b><br/>
  meanBrightness: <span id="meanVal">-</span><br/>
  contrast: <span id="contrastVal">-</span><br/>
  brightnessRange: <span id="rangeVal">-</span><br/>
  prob_light_good: <span id="probVal">-</span><br/>
  torch supported: <span id="torchSup">unknown</span><br/>
  torch state: <span id="torchState">OFF</span>
</div>

<div id="bleStatus">BLE: not connected</div>

<!-- Hidden video + canvas for sending frames to Flask -->
<video id="tagVideo" autoplay playsinline style="display:none;"></video>
<canvas id="tagCanvas" style="display:none;"></canvas>

<script>
(async () => {
  // ==============================
  // 1. MAP + A* NAVIGATION SETUP
  // ==============================
  const canvas = document.getElementById("navCanvas");
  const wrap   = document.getElementById("canvasWrap");
  const ctx    = canvas.getContext("2d");
  const infoEl = document.getElementById("info");
  const poseEl = document.getElementById("pose");
  const setGoalBtn = document.getElementById("setGoal");
  const voiceBtn   = document.getElementById("voiceBtn");
  const permBtn    = document.getElementById("permBtn");

  // BLE UI
  const bleBtn      = document.getElementById("bleConnectBtn");
  const bleStatusEl = document.getElementById("bleStatus");

  // 🔦 Torch-related UI
  const meanSpan     = document.getElementById("meanVal");
  const contrastSpan = document.getElementById("contrastVal");
  const rangeSpan    = document.getElementById("rangeVal");
  const probSpan     = document.getElementById("probVal");
  const torchSupEl   = document.getElementById("torchSup");
  const torchStateEl = document.getElementById("torchState");

  // 🔦 Torch state
  let videoTrack = null;
  let torchSupported = false;
  let torchOn = false;

  async function setTorch(on){
    if (!videoTrack || !torchSupported) return;
    try {
      await videoTrack.applyConstraints({
        advanced: [{ torch: on }]
      });
      torchOn = on;
      torchStateEl.textContent = on ? "ON" : "OFF";
    } catch (err) {
      console.error("Error setting torch:", err);
    }
  }

  // ==========================
  // BLE: ESP32-CAM COMMAND LINK
  // ==========================
  let bleDevice = null;
  let bleServer = null;
  let cmdCharacteristic = null;

  const SERVICE_UUID = '12345678-1234-5678-1234-56789abcdef0';
  const CHARACTERISTIC_UUID = '12345678-1234-5678-1234-56789abcdef1';

  function setBleStatus(msg) {
    if (bleStatusEl) {
      bleStatusEl.textContent = "BLE: " + msg;
    }
  }

  async function connectBLE() {
    try {
      if (!navigator.bluetooth) {
        setBleStatus("Web Bluetooth not supported in this browser.");
        return;
      }
      setBleStatus("Requesting device...");
      bleDevice = await navigator.bluetooth.requestDevice({
        filters: [{ name: 'ESP32-CAM-CMD' }],
        optionalServices: [SERVICE_UUID]
      });

      setBleStatus("Connecting GATT...");
      bleServer = await bleDevice.gatt.connect();

      setBleStatus("Getting service...");
      const service = await bleServer.getPrimaryService(SERVICE_UUID);

      setBleStatus("Getting characteristic...");
      cmdCharacteristic = await service.getCharacteristic(CHARACTERISTIC_UUID);

      setBleStatus("Connected (ready to send commands).");
      if (bleBtn) bleBtn.disabled = true;

      bleDevice.addEventListener("gattserverdisconnected", () => {
        cmdCharacteristic = null;
        setBleStatus("Disconnected (click to reconnect).");
        if (bleBtn) bleBtn.disabled = false;
      });
    } catch (err) {
      console.error(err);
      setBleStatus("Error: " + err.message);
    }
  }

  async function sendBleCommand(text) {
    try {
      if (!cmdCharacteristic) {
        setBleStatus("Not connected to ESP32.");
        return;
      }
      const enc = new TextEncoder();
      const data = enc.encode(text);
      await cmdCharacteristic.writeValue(data);
      setBleStatus("Sent: " + text);
    } catch (err) {
      console.error("sendBleCommand error:", err);
      setBleStatus("Error sending command.");
    }
  }

  if (bleBtn) {
    bleBtn.addEventListener("click", connectBLE);
  }

  // Update torch decision (still uses phone torch)
  function updateTorchDecision(light){
    if (!torchSupported) return;
    if (typeof light.prob_good !== "number") return;

    const p = light.prob_good;
    const ON_TH  = 0.40; // below this → too dark → torch ON

    if (!torchOn && p < ON_TH) {
      setTorch(true);
    }
  }

  // ----- Map config -----
  const m2px_nominal   = 100;   // pixels per meter
  const room_m_width   = 3.9;
  const room_m_height  = 4.8;
  const devicePixelRatio = window.devicePixelRatio || 1;
  const internal_width_px  = room_m_width  * m2px_nominal;
  const internal_height_px = room_m_height * m2px_nominal;

  canvas.width  = internal_width_px * devicePixelRatio;
  canvas.height = internal_height_px * devicePixelRatio;
  canvas.style.width  = Math.min(internal_width_px, window.innerWidth - 40) + "px";
  canvas.style.height = (internal_height_px * (canvas.clientWidth / internal_width_px)) + "px";
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

  const gridSize = 15;
  const m2px = m2px_nominal;

  // ---------- State ----------
  let start = null;    // camera position in room (px)
  let goal = null;
  let mode = null;
  let heading = null;  // compass heading (deg)
  let path = [];
  let lastUtter = null;

  // guidance state machine
  let guidanceBusy = false;     // a guidance cycle is running
  let aligningActive = false;   // waiting for user to rotate
  let alignTargetBearing = null;
  const ALIGN_THRESHOLD_DEG = 15; // degrees
  let lastGuidanceDone = 0;     // timestamp (ms) when last cycle finished
  const GUIDANCE_COOLDOWN_MS = 3000; // 3 s cooldown between cycles
  let alignStartTime = 0;
  const ALIGN_TIMEOUT_MS = 10000; // 5 sec alignment window

  // Obstacles and objects
  const obstacles = [ {x:0, y:0, w:189, h:209},{x:0, y:0, w:19, h:19},{x:0, y:271, w:280, h:209},{x:0, y:460, w:19, h:19},{x:370, y:0, w:19, h:19},{x:313, y:98, w:77, h:214},{x:370, y:460, w:19, h:19},{x:330, y:449, w:40, h:31} ];
  const objects = { chair:{x:345,y:378},dustbin:{x:320,y:63}, bottle:{x:57,y:225},book:{x:123,y:275}, door:{x:284,y:10}};

  // ---------- Utilities ----------
  function info(msg){ infoEl.innerHTML = "Status: " + msg; }
  function clampToRoom(p){
    return {
      x: Math.max(0, Math.min(internal_width_px  - 1, p.x)),
      y: Math.max(0, Math.min(internal_height_px - 1, p.y))
    };
  }
  function toGrid(pt){
    return {
      x: Math.floor(pt.x / gridSize),
      y: Math.floor(pt.y / gridSize)
    };
  }
  function fromGrid(cell){
    return {
      x: cell.x * gridSize + gridSize / 2,
      y: cell.y * gridSize + gridSize / 2
    };
  }

  function safeSpeak(text){
    try {
      if (!("speechSynthesis" in window)) return;
      // cancel any old utterance so we don't overlap
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      u.rate = 1;
      lastUtter = u;
      window.speechSynthesis.speak(u);
    } catch (err) {
      console.error(err);
    }
  }

  // 🔔 Helper: send "haptic" to ESP32 AFTER speech is finished (or fallback to phone vibrate)
  function triggerEspHaptic(pattern) {
    try {
      // If BLE is connected, drive ESP32 LED as "haptic"
      if (cmdCharacteristic) {
        let totalMs = 300;
        if (Array.isArray(pattern) && pattern.length > 0) {
          totalMs = pattern.reduce((s, v) => s + v, 0);
        }
        // Turn LED on, then off after totalMs
        sendBleCommand("LED_ON");
        setTimeout(() => { sendBleCommand("LED_OFF"); }, totalMs);
      } else if ("vibrate" in navigator && pattern) {
        // Fallback: original phone vibration if BLE not connected
        navigator.vibrate(pattern);
      }
    } catch (err) {
      console.error("triggerEspHaptic error:", err);
    }
  }

  // We keep the same name as before so the rest of the code is unchanged.
  function vibrateAfterSpeech(pattern) {
    const synth = window.speechSynthesis;

    function doHaptic() {
      triggerEspHaptic(pattern);
    }

    if (synth && synth.speaking) {
      // Poll until TTS is done, then send haptic
      const check = () => {
        try {
          if (!synth.speaking) {
            doHaptic();
          } else {
            setTimeout(check, 100);
          }
        } catch (err) {
          console.error("vibrateAfterSpeech check error:", err);
        }
      };
      check();
    } else {
      // No speech or already finished
      doHaptic();
    }
  }

  // ---- Steps text helper ----
  function stepsText(dist_m) {
    const stepLen = 0.70; // 70 cm per step
    let rawSteps = dist_m / stepLen;
    let steps = Math.round(rawSteps * 2) / 2;  // nearest 0.5

    if (steps === 0) return "half a step";
    if (steps === 0.5) return "half a step";
    if (steps === 1)   return "one step";
    if (steps === 1.5) return "one and a half steps";

    if (Number.isInteger(steps)) {
      return `${steps} steps`;
    }
    const whole = Math.floor(steps);
    return `${whole} and a half steps`;
  }

  // ---------- Drawing ----------
  function draw(){
    ctx.clearRect(0, 0, internal_width_px, internal_height_px);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, internal_width_px, internal_height_px);

    // Room border
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, internal_width_px, internal_height_px);

    // Obstacles
    ctx.fillStyle = "#666";
    for (const o of obstacles) {
      ctx.fillRect(o.x, o.y, o.w, o.h);
    }

    // Objects
    ctx.fillStyle = "blue";
    ctx.font = "14px Arial";
    for (const k in objects) {
      const p = objects[k];
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(k, p.x + 12, p.y + 6);
    }

    // Start (camera)
    if (start) {
      ctx.fillStyle = "green";
      ctx.beginPath();
      ctx.arc(start.x, start.y, 10, 0, Math.PI * 2);
      ctx.fill();
    }

    // Goal
    if (goal) {
      ctx.fillStyle = "red";
      ctx.beginPath();
      ctx.arc(goal.x, goal.y, 10, 0, Math.PI * 2);
      ctx.fill();
    }

    // Path
    if (path && path.length > 0 && start && goal) {
      ctx.strokeStyle = "blue";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      for (const p of path) {
        const g = fromGrid(p);
        ctx.lineTo(g.x, g.y);
      }
      ctx.stroke();
    }

    // Heading arrow
    if (heading !== null && start) {
      ctx.strokeStyle = "crimson";
      ctx.fillStyle = "crimson";
      ctx.lineWidth = 3;
      const rad = (heading) * Math.PI / 180;
      const vx = Math.sin(rad);
      const vy = -Math.cos(rad);
      const len = 45;
      const tipX = start.x + vx * len;
      const tipY = start.y + vy * len;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();

      const ah = 10;
      const px = -vy;
      const py = vx;
      const bx = tipX - vx * ah;
      const by = tipY - vy * ah;
      const c1x = bx + px * (ah * 0.6);
      const c1y = by + py * (ah * 0.6);
      const c2x = bx - px * (ah * 0.6);
      const c2y = by - py * (ah * 0.6);
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(c1x, c1y);
      ctx.lineTo(c2x, c2y);
      ctx.closePath();
      ctx.fill();
    }
  }

  // ---------- Grid ----------
  function makeGrid(){
    const rows = Math.ceil(internal_height_px / gridSize);
    const cols = Math.ceil(internal_width_px / gridSize);
    const grid = Array.from({ length: rows }, () => Array(cols).fill(0));

    for (const o of obstacles) {
      const x0 = Math.floor(o.x / gridSize);
      const y0 = Math.floor(o.y / gridSize);
      const x1 = Math.floor((o.x + o.w) / gridSize);
      const y1 = Math.floor((o.y + o.h) / gridSize);
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          if (y >= 0 && x >= 0 && y < rows && x < cols) {
            grid[y][x] = 1;
          }
        }
      }
    }
    return grid;
  }

  // (optional) helper to inspect number of turns
  function countTurns(pathCells){
    if(!pathCells || pathCells.length < 2) return 0;
    let turns = 0;
    let dx0 = pathCells[1].x - pathCells[0].x;
    let dy0 = pathCells[1].y - pathCells[0].y;
    for(let i=2;i<pathCells.length;i++){
      const dxi = pathCells[i].x - pathCells[i-1].x;
      const dyi = pathCells[i].y - pathCells[i-1].y;
      if(dxi !== dx0 || dyi !== dy0){
        turns++;
        dx0 = dxi;
        dy0 = dyi;
      }
    }
    return turns;
  }

  // 🔄 Turn-only Dijkstra-like search
  function astar(grid, startCell, goalCell){
    const rows = grid.length;
    const cols = grid[0].length;
    const sk = (x,y,dx,dy) => `${x}_${y}_${dx}_${dy}`;

    const open = new Map();
    const closed = new Set();

    // Node: { x, y, dx, dy, turns, steps, f, parent }
    const startKey = sk(startCell.x, startCell.y, 0, 0);
    open.set(startKey, {
      x: startCell.x,
      y: startCell.y,
      dx: 0,
      dy: 0,
      turns: 0,
      steps: 0,
      f: 0,
      parent: null
    });

    const neighbors = [
      [ 1,  0],
      [-1,  0],
      [ 0,  1],
      [ 0, -1]
    ];

    while(open.size){
      let bestKey = null;
      let best = null;
      for(const [k,v] of open){
        if(!best || v.f < best.f){
          best = v;
          bestKey = k;
        }
      }
      if(!best) break;

      if(best.x === goalCell.x && best.y === goalCell.y){
        const out = [];
        let n = best;
        while(n){
          out.push({x:n.x, y:n.y});
          n = n.parent;
        }
        return out.reverse();
      }

      open.delete(bestKey);
      closed.add(bestKey);

      for(const [mdx,mdy] of neighbors){
        const nx = best.x + mdx;
        const ny = best.y + mdy;
        if(nx<0 || ny<0 || nx>=cols || ny>=rows) continue;
        if(grid[ny][nx] === 1) continue; // obstacle

        let isTurn = false;
        if(!(best.dx === 0 && best.dy === 0)){
          const sameDir = (best.dx === mdx && best.dy === mdy);
          if(!sameDir) isTurn = true;
        }
        const turns2 = best.turns + (isTurn ? 1 : 0);
        const steps2 = best.steps + 1;

        const key2 = sk(nx,ny,mdx,mdy);
        if(closed.has(key2)) continue;

        const f2 = turns2; // only turns matter

        const exist = open.get(key2);
        if(
          !exist ||
          (turns2 < exist.turns) ||
          (turns2 === exist.turns && steps2 < exist.steps)
        ){
          open.set(key2, {
            x: nx,
            y: ny,
            dx: mdx,
            dy: mdy,
            turns: turns2,
            steps: steps2,
            f: f2,
            parent: best
          });
        }
      }
    }
    return null;
  }

  // helper: end of one guidance cycle
  function endGuidanceCycle(){
    guidanceBusy = false;
    aligningActive = false;
    alignTargetBearing = null;
    lastGuidanceDone = Date.now();
  }

  // ---------- Guidance computation ----------
  function computeGuidanceAndSpeak(){
    try {
      // At the start of each cycle, clear alignment flags
      aligningActive = false;
      alignTargetBearing = null;

      if (!start || !goal) {
        info("Waiting for camera start and goal.");
        endGuidanceCycle();
        return;
      }

      // 1) DESTINATION REACHED CHECK (30 cm)
      const dxGoal_px = goal.x - start.x;
      const dyGoal_px = goal.y - start.y;
      const distToGoal_m = Math.hypot(dxGoal_px, dyGoal_px) / m2px;
      if (!isFinite(distToGoal_m)) {
        info("Invalid start/goal position.");
        endGuidanceCycle();
        return;
      }
      if (distToGoal_m <= 0.28) {
        info("✅ Destination reached. Please set a new goal.");
        safeSpeak("Destination reached. Please select a new destination.");
        vibrateAfterSpeech([300,150,300]);
        setTorch(false);
        //  RESET SYSTEM FOR NEXT CYCLE
        goal = null;
        path = [];
        aligningActive = false;
        alignTargetBearing = null;
        draw();
        endGuidanceCycle();
        return;
      }

      // 2) Build grid and path
      const grid = makeGrid();
      const sCell = toGrid(start);
      const gCell = toGrid(goal);

      if (!Number.isFinite(sCell.x) || !Number.isFinite(sCell.y) ||
          !Number.isFinite(gCell.x) || !Number.isFinite(gCell.y)) {
        info("Invalid grid cell index (NaN).");
        endGuidanceCycle();
        return;
      }

      // goal inside obstacle → move to nearest free cell
      if (grid[gCell.y] && grid[gCell.y][gCell.x] === 1) {
        info("Goal is inside an obstacle — searching nearest free cell...");
        const rows = grid.length;
        const cols = grid[0].length;
        let found = null;
        let minDist = Infinity;
        for (let y = 0; y < rows; y++) {
          for (let x = 0; x < cols; x++) {
            if (grid[y][x] === 0) {
              const dx = x - gCell.x;
              const dy = y - gCell.y;
              const d = Math.sqrt(dx * dx + dy * dy);
              if (d < minDist) {
                minDist = d;
                found = { x, y };
              }
            }
          }
        }
        if (found) {
          goal = fromGrid(found);
          gCell.x = found.x;
          gCell.y = found.y;
          info("Goal adjusted to nearest reachable cell.");
        } else {
          info("No reachable area found in map.");
          endGuidanceCycle();
          return;
        }
      }

      if (grid[sCell.y] && grid[sCell.y][sCell.x] === 1) {
        info("Start overlaps obstacle — move tag or adjust map.");
        endGuidanceCycle();
        return;
      }

      const newPath = astar(grid, sCell, gCell);
      if (!newPath) {
        info("No path found from start to goal.");
        path = [];
        draw();
        endGuidanceCycle();
        return;
      }
      path = newPath;

      if (path.length <= 1) {
        draw();
        endGuidanceCycle();
        return;
      }

      // Straight-line segment until next turn
      let dx0 = path[1].x - path[0].x;
      let dy0 = path[1].y - path[0].y;
      let turnIdx = 1;
      for (let i = 2; i < path.length; i++) {
        const dxi = path[i].x - path[i-1].x;
        const dyi = path[i].y - path[i-1].y;
        if (dxi !== dx0 || dyi !== dy0) break;
        turnIdx = i;
      }

      const startPt = fromGrid(path[0]);
      const turnPt  = fromGrid(path[turnIdx]);
      const dist_m  = Math.sqrt(
                        (turnPt.x - startPt.x) ** 2 +
                        (turnPt.y - startPt.y) ** 2
                      ) / m2px;

      const stepText = stepsText(dist_m);

      // Compute target bearing (deg)
      let targetBearing = Math.atan2(
        turnPt.x - startPt.x,
        -(turnPt.y - startPt.y)
      ) * 180 / Math.PI;
      targetBearing = (targetBearing + 360) % 360;

      if (typeof heading !== "number") {
        const msg = `Walk about ${stepText} straight ahead (compass unavailable).`;
        info("Next: " + msg);
        safeSpeak(msg);
        // still give haptic after speech to confirm
        vibrateAfterSpeech([200, 100, 200]);
        draw();
        endGuidanceCycle();
        return;
      }

      const normalize = a => (a + 360) % 360;
      const signedDiff = (t, u) => {
        let d = normalize(t) - normalize(u);
        if (d > 180) d -= 360;
        if (d <= -180) d += 360;
        return d;
      };

      const delta = signedDiff(targetBearing, heading);
      const absd = Math.abs(delta);

      let msg = "";

      if (absd < ALIGN_THRESHOLD_DEG) {
        // Already aligned: just walk
        msg = `Go ahead and walk about ${stepText}.`;
        info("Next: " + msg);
        safeSpeak(msg);
        // haptic strictly AFTER speech, two pulses
        vibrateAfterSpeech([200, 100, 200]);
        draw();
        endGuidanceCycle();
      } else {
        // Need to turn; speak once, then alignment handled by orientation events
        const dir = (delta > 0) ? "right" : "left";
        msg = `Turn ${Math.round(absd)} degrees ${dir} and walk about ${stepText} ahead.`;
        info("Next: " + msg);
        safeSpeak(msg);
        aligningActive = true;
        alignTargetBearing = targetBearing;
        alignStartTime = Date.now(); // mark alignment start time (cooldown-style logic)

        draw();
        // guidanceBusy stays true; endGuidanceCycle will be called when alignment is done
      }

    } catch (err) {
      console.error("computeGuidanceAndSpeak exception:", err);
      info("Error computing guidance — check console.");
      endGuidanceCycle();
    }
  }

  // ---------- Voice commands (goal selection) ----------
  function voiceCommand(){
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      info("Voice recognition not supported.");
      return;
    }
    const rec = new SpeechRecognition();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.onstart = () => info("🎙️ Listening...");
    rec.onerror = e => info("Voice error: " + e.error);
    rec.onresult = (e) => {
   let matched = false;

   console.log("FULL RESULT:", e.results);

   //  Check ALL alternatives
   for (let i = 0; i < e.results[0].length; i++) {
   const text = e.results[0][i].transcript.toLowerCase();
   console.log("Heard alt:", text);

   const synonyms = {
    chair: ["chair", "share", "cheer", "char", "care", "air"],
    bottle: ["bottle"],
    book: ["book"],
    door: ["door"],
    dustbin: ["dustbin", "bin"]
   };

    for (const key in synonyms) {
    for (const word of synonyms[key]) {
      if (text.includes(word)) {
        goal = { x: objects[key].x, y: objects[key].y };
        info(" Goal set to " + key + " (heard: " + text + ")");
        draw();
        matched = true;
        return;
      }
    }
  }
}

if (!matched) {
 info("Object not recognized. Say: book, bottle, table or door.");
}
    };
    rec.start();
  }

  // ---------- Orientation / Compass ----------
  async function requestIOSPermissionIfNeeded(){
    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function') {
      permBtn.style.display = 'inline-block';
      permBtn.onclick = async () => {
        const perm = await DeviceOrientationEvent.requestPermission();
        if (perm === 'granted') {
          permBtn.style.display = 'none';
          info("Compass permission granted.");
        } else {
          info("Compass permission denied.");
        }
      };
    } else {
      permBtn.style.display = 'none';
    }
  }

  function attachOrientationListeners(){
    const normalize = a => (a + 360) % 360;

    window.addEventListener(
      'deviceorientationabsolute',
      (e) => {
        try {
          if (typeof e.alpha === 'number') {
            // Simple mapping, you can tweak if needed
            heading = normalize(85 - (e.alpha));
            draw();

            // Alignment phase: only active after a "turn" instruction
            if (aligningActive && alignTargetBearing !== null && guidanceBusy) {
                 const signedDiff = (t, u) => {
                   let d = (t + 360) % 360 - (u + 360) % 360;
                   if (d > 180) d -= 360;
                   if (d <= -180) d += 360;
                   return d;
                   };
               
                 const delta = signedDiff(alignTargetBearing, heading);
                 const absd = Math.abs(delta);
               
                 //  SUCCESS: aligned
                 if (absd < ALIGN_THRESHOLD_DEG) {
                   aligningActive = false;
                   alignTargetBearing = null;
               
                   vibrateAfterSpeech([200, 100, 200]);
                   endGuidanceCycle();
                   return;
                  }
               
                 // COOLDOWN-STYLE ALIGNMENT TIME CHECK (NO setTimeout)
                 const now = Date.now();
               
                 if (now - alignStartTime > ALIGN_TIMEOUT_MS) {
                   console.log("⏱ Alignment timeout → recomputing guidance");
               
                   aligningActive = false;
                   alignTargetBearing = null;
               
                   // 🔁 restart full guidance cycle (Option A)
                   computeGuidanceAndSpeak();
                  }
             }
          }
        } catch (err) {
          console.error("deviceorientation handler error:", err);
        }
      },
      true
    );
  }

  function clientToInternal(clientX, clientY){
    const rect = canvas.getBoundingClientRect();
    const sx = internal_width_px  / rect.width;
    const sy = internal_height_px / rect.height;
    return {
      x: (clientX - rect.left) * sx,
      y: (clientY - rect.top)  * sy
    };
  }

  function onMapPointer(x, y){
    const p = clampToRoom({ x, y });
    if (mode === "goal") {
      goal = p;
      mode = null;
      info("Goal set.");
      draw();
      if (!start) info("Waiting for camera start from AprilTag...");
    }
  }

  canvas.addEventListener(
    'click',
    (ev) => {
      const p = clientToInternal(ev.clientX, ev.clientY);
      onMapPointer(p.x, p.y);
    },
    false
  );
  canvas.addEventListener(
    'touchstart',
    (ev) => {
      ev.preventDefault();
      const t = ev.touches[0];
      const p = clientToInternal(t.clientX, t.clientY);
      onMapPointer(p.x, p.y);
    },
    { passive: false }
  );

  setGoalBtn.onclick = () => {
    mode = "goal";
    info("Tap the map to set GOAL position.");
  };
  voiceBtn.onclick = voiceCommand;

  draw();
  info("Ready. Waiting for AprilTag camera position. Set goal by tap or voice.");

  await requestIOSPermissionIfNeeded();
  attachOrientationListeners();

  window.addEventListener('beforeunload', () => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  });

  // ==============================
  // 2. STILLNESS DETECTION (ACCELEROMETER)
  // ==============================
  let accHistory = [];
  let wasStill = false;
  const ACC_WINDOW_SAMPLES = 10;
  const ACC_STILL_THRESHOLD = 0.28;   // m/s^2  (tune if needed)

  function updateStillFromAccel(ax, ay, az) {
    try {
      const mag = Math.sqrt(ax*ax + ay*ay + az*az);
      accHistory.push(mag);
      if (accHistory.length > ACC_WINDOW_SAMPLES) {
        accHistory.shift();
      }

      if (accHistory.length < 3) {
        wasStill = false;
        return;
      }

      const avg = accHistory.reduce((s, v) => s + v, 0) / accHistory.length;
      const isStill = avg < ACC_STILL_THRESHOLD;

      // MOVING -> STILL edge
      if (isStill && !wasStill) {
        const now = Date.now();
        if (
          start && goal &&
          !guidanceBusy &&                        // don't re-enter
          (now - lastGuidanceDone > GUIDANCE_COOLDOWN_MS) // 3s cooldown
        ) {
          guidanceBusy = true;                   // start new cycle
          computeGuidanceAndSpeak();             // full cycle: speak, align, haptic, cooldown
        }
      }

      wasStill = isStill;
    } catch (err) {
      console.error("updateStillFromAccel error:", err);
      wasStill = false;
    }
  }

  window.addEventListener("devicemotion", (e) => {
    try {
      const a = e.acceleration;
      if (!a || a.x == null || a.y == null || a.z == null) return;
      updateStillFromAccel(a.x, a.y, a.z);
    } catch (err) {
      console.error("devicemotion handler error:", err);
    }
  });

  // ==============================
  // 3. APRILTAG VIDEO + /frame LOOP + TORCH
  // ==============================
  const tagVideo  = document.getElementById("tagVideo");
  const tagCanvas = document.getElementById("tagCanvas");
  const tagCtx    = tagCanvas.getContext("2d");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:  { ideal: 1080 },
        height: { ideal: 1080 },
        frameRate: { ideal: 3, max: 3 },
        facingMode: { exact: "environment" }
      },
      audio: false
    });
    tagVideo.srcObject = stream;
    await tagVideo.play();
    info("Camera started. Waiting for tag to set start position.");

    // 🔦 Torch capability detection
    const tracks = stream.getVideoTracks();
    if (tracks.length > 0) {
      videoTrack = tracks[0];
      const caps = videoTrack.getCapabilities ? videoTrack.getCapabilities() : {};
      if (caps && ("torch" in caps)) {
        torchSupported = true;
        torchSupEl.textContent = "YES";
      } else {
        torchSupported = false;
        torchSupEl.textContent = "NO";
      }
    } else {
      torchSupEl.textContent = "NO TRACK";
    }

  } catch (e) {
    console.error(e);
    info("Camera error: " + e.message);
    return;
  }

  async function sendFrame(){
    try {
      const w = tagVideo.videoWidth;
      const h = tagVideo.videoHeight;
      if (!w || !h) return;

      tagCanvas.width  = w;
      tagCanvas.height = h;
      tagCtx.drawImage(tagVideo, 0, 0, w, h);

      tagCanvas.toBlob(async (blob) => {
        try {
          if (!blob) return;
          const reader = new FileReader();
          reader.onloadend = async () => {
            try {
              const base64data = reader.result.split(",")[1];
              const res = await fetch("/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: base64data })
              });
              const data = await res.json();

              // ----- 🔦 Light info from backend -----
              if (data.light) {
                const lf = data.light;
                if (typeof lf.meanBrightness === "number") {
                  meanSpan.textContent = lf.meanBrightness.toFixed(2);
                }
                if (typeof lf.contrast === "number") {
                  contrastSpan.textContent = lf.contrast.toFixed(2);
                }
                if (typeof lf.brightnessRange === "number") {
                  rangeSpan.textContent = lf.brightnessRange.toFixed(2);
                }
                if (typeof lf.prob_good === "number") {
                  probSpan.textContent = lf.prob_good.toFixed(3);
                }

                updateTorchDecision(lf);
              }

              if (!data.detections || data.detections.length === 0) {
                poseEl.textContent = "Camera pose: no good tag detection.";
                return;
              }

              const d = data.detections[0];
              const ct = d.cam_in_tag;
              const cr = d.cam_in_room;

              poseEl.textContent =
                "Tag id: " + d.id + "\\n" +
                "Camera in TAG frame (m): [" + ct.map(v => v.toFixed(4)).join(", ") + "]\\n" +
                "Camera in ROOM frame (m): [" + cr.map(v => v.toFixed(4)).join(", ")+ "]";

              const camX_m = cr[0];
              const camY_m = cr[1];

              let px = camX_m * m2px;
              let py = camY_m * m2px;
              const p = clampToRoom({ x: px, y: py });

              start = p;
              draw();
            } catch (err) {
              console.error("sendFrame inner reader error:", err);
            }
          };
          reader.readAsDataURL(blob);
        } catch (err) {
          console.error("sendFrame blob error:", err);
        }
      }, "image/jpeg", 0.7);
    } catch (err) {
      console.error("sendFrame error:", err);
    }
  }

  setInterval(sendFrame, 333);  // ~3 FPS

})();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/frame", methods=["POST"])
def frame():
    global tag_positions, last_tag_id

    j = request.get_json()
    if not j or "image" not in j:
        return jsonify({"error": "no image"}), 400

    try:
        img_data = base64.b64decode(j["image"])
    except Exception:
        return jsonify({"error": "bad base64"}), 400

    np_data = np.frombuffer(img_data, np.uint8)
    gray = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return jsonify({"error": "decode failed"}), 400

    # ----- 🔦 Brightness features (always, even if no tag) -----
    gray_small = cv2.resize(gray, (320, 320))
    feats = compute_brightness_features(gray_small)
    p_good = light_prob_good(
        feats["meanBrightness"],
        feats["contrast"],
        feats["brightnessRange"]
    )
    light_info = {
        "meanBrightness": feats["meanBrightness"],
        "contrast": feats["contrast"],
        "brightnessRange": feats["brightnessRange"],
        "prob_good": p_good
    }

    # ----- Undistort for tag detection -----
    undistorted = cv2.undistort(gray, K, distCoeffs)

    detections = detector.detect(
        undistorted,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
    )

    if not detections:
        return jsonify({
            "detections": [],
            "light": light_info
        })

    MAX_POSE_ERR = 1.0
    MIN_MARGIN = 30.0

    best = None
    best_err = None

    for det in detections:
        pose_err = getattr(det, "pose_err", None)
        if pose_err is None:
            continue
        if det.decision_margin < MIN_MARGIN:
            continue
        if best is None or pose_err < best_err:
            best = det
            best_err = pose_err

    if best is None or best_err is None or best_err > MAX_POSE_ERR:
        return jsonify({
            "detections": [],
            "light": light_info
        })

    # Reset buffer when tag id changes
    if last_tag_id is None or int(best.tag_id) != int(last_tag_id):
        tag_positions = []
        last_tag_id = int(best.tag_id)

    R = best.pose_R
    t = best.pose_t.reshape(3, 1)

    R_T = R.T
    cam_in_tag = - R_T @ t
    cam_in_tag_vec = cam_in_tag.reshape(3)

    tag_positions.append(cam_in_tag_vec)
    if len(tag_positions) > ROOM_BUFFER_SIZE:
        tag_positions = tag_positions[-ROOM_BUFFER_SIZE:]

    tag_median = median_position(tag_positions)
    if tag_median is None:
        tag_median = cam_in_tag_vec

    cam_in_room_median = camera_pos_in_room(best.tag_id, tag_median)
    if cam_in_room_median is None:
        cam_in_room_median = np.array(
            [np.nan, np.nan, np.nan],
            dtype=np.float64
        )

    cam_in_tag_list = cam_in_tag_vec.tolist()
    cam_in_room_list = cam_in_room_median.tolist()

    return jsonify({
        "detections": [{
            "id": int(best.tag_id),
            "cam_in_tag": cam_in_tag_list,
            "cam_in_room": cam_in_room_list
        }],
        "light": light_info
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
