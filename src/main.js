import "./style.css";
import { FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";

const ASSET_BASE = import.meta.env.BASE_URL;
const WASM_ROOT = `${ASSET_BASE}vendor/mediapipe/wasm`;
const MODEL_PATH = `${ASSET_BASE}models/gesture_recognizer.task`;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

const startButton = document.querySelector("#startButton");
const resetButton = document.querySelector("#resetButton");
const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");

const repCountEl = document.querySelector("#repCount");
const phaseEl = document.querySelector("#phase");
const stableGestureEl = document.querySelector("#stableGesture");
const gestureScoreEl = document.querySelector("#gestureScore");
const cosineSimilarityEl = document.querySelector("#cosineSimilarity");
const opennessValueEl = document.querySelector("#opennessValue");
const normalizedValueEl = document.querySelector("#normalizedValue");
const opennessBarEl = document.querySelector("#opennessBar");
const normalizedBarEl = document.querySelector("#normalizedBar");
const eventLogEl = document.querySelector("#eventLog");
const statusPillEl = document.querySelector("#statusPill");

let gestureRecognizer;
let webcamStream;
let rafId;
let lastVideoTime = -1;

const appState = {
  repCount: 0,
  phase: "等待张开",
  stableGesture: "-",
  gestureScore: 0,
  cosineSimilarity: 0,
  openness: 0,
  normalizedOpenness: 0,
  modelReady: false
};

const gestureStabilizer = {
  stableLabel: "No_Hand",
  candidateLabel: null,
  streak: 0,
  thresholdFrames: 3,
  update(nextLabel) {
    if (nextLabel === this.candidateLabel) {
      this.streak += 1;
    } else {
      this.candidateLabel = nextLabel;
      this.streak = 1;
    }

    if (this.streak >= this.thresholdFrames) {
      this.stableLabel = nextLabel;
    }

    return this.stableLabel;
  },
  reset() {
    this.stableLabel = "No_Hand";
    this.candidateLabel = null;
    this.streak = 0;
  }
};

const opennessTracker = {
  emaValue: null,
  min: Infinity,
  max: -Infinity,
  alpha: 0.25,
  update(rawValue) {
    if (this.emaValue === null) {
      this.emaValue = rawValue;
    } else {
      this.emaValue = this.alpha * rawValue + (1 - this.alpha) * this.emaValue;
    }

    this.min = Math.min(this.min, this.emaValue);
    this.max = Math.max(this.max, this.emaValue);

    const range = Math.max(0.001, this.max - this.min);
    const normalized = Math.min(
      1,
      Math.max(0, (this.emaValue - this.min) / range)
    );

    return {
      openness: this.emaValue,
      normalized,
      range
    };
  },
  reset() {
    this.emaValue = null;
    this.min = Infinity;
    this.max = -Infinity;
  }
};

const repetitionEngine = {
  repCount: 0,
  minRange: 0.3,
  minAmplitude: 0.34,
  similarityThreshold: 0.92,
  derivativeEpsilon: 0.012,
  reversalConfirmDelta: 0.07,
  minKeyframeGapMs: 140,
  minStableHoldMs: 120,
  minCycleMs: 500,
  maxCycleMs: 5000,
  cooldownMs: 260,
  samples: [],
  anchorKeyframe: null,
  middleKeyframe: null,
  currentTrend: 0,
  extremeCandidate: null,
  candidateStartedAtMs: -Infinity,
  lastKeyframeTs: -Infinity,
  lastRepTs: -Infinity,
  lastSimilarity: 0,
  reset() {
    this.repCount = 0;
    this.samples = [];
    this.anchorKeyframe = null;
    this.middleKeyframe = null;
    this.currentTrend = 0;
    this.extremeCandidate = null;
    this.candidateStartedAtMs = -Infinity;
    this.lastKeyframeTs = -Infinity;
    this.lastRepTs = -Infinity;
    this.lastSimilarity = 0;
  },
  update({ normalized, range, timestampMs, poseVector }) {
    if (!Number.isFinite(normalized) || !poseVector) {
      return {
        repAdded: false,
        phaseLabel: this.phaseLabel(),
        similarity: this.lastSimilarity
      };
    }

    this.samples.push({
      timestampMs,
      value: normalized,
      vector: poseVector
    });

    if (this.samples.length > 6) {
      this.samples.shift();
    }

    if (range < this.minRange) {
      return {
        repAdded: false,
        phaseLabel: "建立关键帧基线中",
        similarity: this.lastSimilarity
      };
    }

    const keyframe = this.detectKeyframe();
    if (!keyframe) {
      return {
        repAdded: false,
        phaseLabel: this.phaseLabel(),
        similarity: this.lastSimilarity
      };
    }

    return this.processKeyframe(keyframe);
  },
  detectKeyframe() {
    if (this.samples.length < 2) {
      return null;
    }

    const previous = this.samples.at(-2);
    const current = this.samples.at(-1);
    const movement = signWithDeadband(
      current.value - previous.value,
      this.derivativeEpsilon
    );

    if (!this.extremeCandidate) {
      this.extremeCandidate = previous;
      this.candidateStartedAtMs = previous.timestampMs;
    }

    if (this.currentTrend === 0) {
      if (movement === 0) {
        return null;
      }

      const stableHoldMs = current.timestampMs - this.candidateStartedAtMs;
      if (
        stableHoldMs >= this.minStableHoldMs &&
        this.extremeCandidate.timestampMs - this.lastKeyframeTs >= this.minKeyframeGapMs
      ) {
        const inferredType = movement < 0 ? "peak" : "valley";
        const keyframe = {
          type: inferredType,
          value: this.extremeCandidate.value,
          timestampMs: this.extremeCandidate.timestampMs,
          vector: this.extremeCandidate.vector
        };

        this.lastKeyframeTs = this.extremeCandidate.timestampMs;
        this.currentTrend = movement;
        this.extremeCandidate = current;
        this.candidateStartedAtMs = current.timestampMs;
        return keyframe;
      }

      this.currentTrend = movement;
      this.extremeCandidate = current;
      this.candidateStartedAtMs = current.timestampMs;
      return null;
    }

    if (this.currentTrend > 0) {
      if (movement >= 0) {
        if (current.value >= this.extremeCandidate.value) {
          this.extremeCandidate = current;
          this.candidateStartedAtMs = current.timestampMs;
        }
        return null;
      }

      if (
        this.extremeCandidate.value - current.value <
        this.reversalConfirmDelta
      ) {
        return null;
      }

      if (
        this.extremeCandidate.timestampMs - this.lastKeyframeTs <
        this.minKeyframeGapMs
      ) {
        this.currentTrend = movement;
        this.extremeCandidate = current;
        this.candidateStartedAtMs = current.timestampMs;
        return null;
      }

      const keyframe = {
        type: "peak",
        value: this.extremeCandidate.value,
        timestampMs: this.extremeCandidate.timestampMs,
        vector: this.extremeCandidate.vector
      };

      this.lastKeyframeTs = this.extremeCandidate.timestampMs;
      this.currentTrend = movement;
      this.extremeCandidate = current;
      this.candidateStartedAtMs = current.timestampMs;
      return keyframe;
    }

    if (movement <= 0) {
      if (current.value <= this.extremeCandidate.value) {
        this.extremeCandidate = current;
        this.candidateStartedAtMs = current.timestampMs;
      }
      return null;
    }

    if (
      current.value - this.extremeCandidate.value <
      this.reversalConfirmDelta
    ) {
      return null;
    }

    if (
      this.extremeCandidate.timestampMs - this.lastKeyframeTs <
      this.minKeyframeGapMs
    ) {
      this.currentTrend = movement;
      this.extremeCandidate = current;
      this.candidateStartedAtMs = current.timestampMs;
      return null;
    }

    const keyframe = {
      type: "valley",
      value: this.extremeCandidate.value,
      timestampMs: this.extremeCandidate.timestampMs,
      vector: this.extremeCandidate.vector
    };

    this.lastKeyframeTs = this.extremeCandidate.timestampMs;
    this.currentTrend = movement;
    this.extremeCandidate = current;
    this.candidateStartedAtMs = current.timestampMs;
    return keyframe;
  },
  processKeyframe(keyframe) {
    if (!this.anchorKeyframe) {
      this.anchorKeyframe = keyframe;
      return {
        repAdded: false,
        phaseLabel: this.phaseLabel(),
        similarity: this.lastSimilarity,
        keyframe
      };
    }

    if (!this.middleKeyframe) {
      if (keyframe.type === this.anchorKeyframe.type) {
        this.anchorKeyframe = keyframe;
        return {
          repAdded: false,
          phaseLabel: this.phaseLabel(),
          similarity: this.lastSimilarity,
          keyframe
        };
      }

      const amplitude = Math.abs(keyframe.value - this.anchorKeyframe.value);
      if (amplitude >= this.minAmplitude) {
        this.middleKeyframe = keyframe;
      }

      return {
        repAdded: false,
        phaseLabel: this.phaseLabel(),
        similarity: this.lastSimilarity,
        keyframe
      };
    }

    if (keyframe.type !== this.anchorKeyframe.type) {
      const amplitude = Math.abs(keyframe.value - this.anchorKeyframe.value);
      if (amplitude >= this.minAmplitude) {
        this.middleKeyframe = keyframe;
      }

      return {
        repAdded: false,
        phaseLabel: this.phaseLabel(),
        similarity: this.lastSimilarity,
        keyframe
      };
    }

    const cycleDuration = keyframe.timestampMs - this.anchorKeyframe.timestampMs;
    const firstAmplitude = Math.abs(this.middleKeyframe.value - this.anchorKeyframe.value);
    const secondAmplitude = Math.abs(keyframe.value - this.middleKeyframe.value);
    const similarity = cosineSimilarity(this.anchorKeyframe.vector, keyframe.vector);
    this.lastSimilarity = similarity;

    const canCount =
      similarity >= this.similarityThreshold &&
      firstAmplitude >= this.minAmplitude &&
      secondAmplitude >= this.minAmplitude &&
      cycleDuration >= this.minCycleMs &&
      cycleDuration <= this.maxCycleMs &&
      keyframe.timestampMs - this.lastRepTs >= this.cooldownMs;

    this.anchorKeyframe = keyframe;
    this.middleKeyframe = null;

    if (canCount) {
      this.repCount += 1;
      this.lastRepTs = keyframe.timestampMs;
      return {
        repAdded: true,
        phaseLabel: this.phaseLabel(),
        similarity,
        keyframe
      };
    }

    return {
      repAdded: false,
      phaseLabel: similarity < this.similarityThreshold
        ? "关键帧回归不够像，重新建立起点"
        : this.phaseLabel(),
      similarity,
      keyframe
    };
  },
  phaseLabel() {
    if (!this.anchorKeyframe) {
      return "等待首个关键帧";
    }

    if (!this.middleKeyframe) {
      return this.anchorKeyframe.type === "peak"
        ? "已捕获峰值，等待谷值"
        : "已捕获谷值，等待峰值";
    }

    return this.anchorKeyframe.type === "peak"
      ? "已形成峰谷，等待返回峰值"
      : "已形成谷峰，等待返回谷值";
  }
};

async function initializeRecognizer() {
  updateStatus("加载手势模型...");

  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_PATH
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
    cannedGesturesClassifierOptions: {
      scoreThreshold: 0.55
    }
  });

  appState.modelReady = true;
  updateStatus("模型已就绪");
}

async function startCamera() {
  if (!appState.modelReady) {
    await initializeRecognizer();
  }

  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: false
  });

  video.srcObject = webcamStream;
  await video.play();

  resizeCanvas();
  startButton.disabled = true;
  startButton.textContent = "识别中";
  logEvent("摄像头已启动，开始检测手势。");

  renderLoop();
}

function renderLoop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const nowMs = performance.now();
    const result = gestureRecognizer.recognizeForVideo(video, nowMs);
    processFrame(result, nowMs);
  }

  rafId = requestAnimationFrame(renderLoop);
}

function processFrame(result, timestampMs) {
  syncCanvasToVideo();
  clearCanvas();

  const landmarks = result.landmarks?.[0];
  const worldLandmarks = result.worldLandmarks?.[0];
  const gestureCategory = result.gestures?.[0]?.[0];
  const rawGesture = landmarks && gestureCategory ? gestureCategory.categoryName : "No_Hand";
  const stableGesture = gestureStabilizer.update(rawGesture);
  const gestureScore = gestureCategory?.score ?? 0;

  appState.stableGesture = stableGesture === "No_Hand" ? "-" : stableGesture;
  appState.gestureScore = gestureScore;

  if (!landmarks) {
    renderUI({
      openness: 0,
      normalized: 0,
      phaseLabel: repetitionEngine.phaseLabel()
    });
    return;
  }

  drawHand(landmarks);

  const rawOpenness = computeHandOpenness(landmarks);
  const poseVector = computePoseVector(worldLandmarks ?? landmarks);
  const opennessState = opennessTracker.update(rawOpenness);
  const repetition = repetitionEngine.update({
    normalized: opennessState.normalized,
    range: opennessState.range,
    timestampMs,
    poseVector
  });

  if (repetition.repAdded) {
    logEvent(
      `计数 +1，总计 ${repetitionEngine.repCount} 次，关键帧余弦 ${(repetition.similarity * 100).toFixed(1)}%。`
    );
  }

  renderUI({
    openness: opennessState.openness,
    normalized: opennessState.normalized,
    phaseLabel: repetition.phaseLabel,
    similarity: repetition.similarity
  });
}

function computeHandOpenness(landmarks) {
  const wrist = landmarks[0];
  const palmAnchor = landmarks[9];
  const palmScale = distance(wrist, palmAnchor) || 0.001;
  const fingertipIndices = [4, 8, 12, 16, 20];

  const averageDistance =
    fingertipIndices.reduce((sum, index) => sum + distance(wrist, landmarks[index]), 0) /
    fingertipIndices.length;

  return averageDistance / palmScale;
}

function computePoseVector(landmarks) {
  if (!landmarks?.length) {
    return null;
  }

  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = distance(wrist, middleMcp) || 0.001;
  const vector = [];

  for (const point of landmarks) {
    vector.push((point.x - wrist.x) / scale);
    vector.push((point.y - wrist.y) / scale);
    vector.push(((point.z ?? 0) - (wrist.z ?? 0)) / scale);
  }

  return vector;
}

function renderUI({ openness, normalized, phaseLabel, similarity = 0 }) {
  appState.repCount = repetitionEngine.repCount;
  appState.phase = phaseLabel;
  appState.openness = openness;
  appState.normalizedOpenness = normalized;
  appState.cosineSimilarity = similarity;

  repCountEl.textContent = String(appState.repCount);
  phaseEl.textContent = appState.phase;
  stableGestureEl.textContent = prettyGesture(appState.stableGesture);
  gestureScoreEl.textContent =
    appState.gestureScore > 0 ? `${(appState.gestureScore * 100).toFixed(0)}%` : "-";
  cosineSimilarityEl.textContent =
    similarity > 0 ? `${(similarity * 100).toFixed(1)}%` : "-";
  opennessValueEl.textContent = openness.toFixed(2);
  normalizedValueEl.textContent = normalized.toFixed(2);
  opennessBarEl.style.width = `${Math.min(openness / 3, 1) * 100}%`;
  normalizedBarEl.style.width = `${normalized * 100}%`;
}

function prettyGesture(label) {
  const mapping = {
    Open_Palm: "张开手掌",
    Closed_Fist: "握拳",
    Pointing_Up: "食指指向",
    Thumb_Up: "点赞",
    Thumb_Down: "倒赞",
    Victory: "剪刀手",
    ILoveYou: "手语 ILY",
    "-": "-"
  };
  return mapping[label] ?? label;
}

function drawHand(landmarks) {
  ctx.save();
  ctx.strokeStyle = "rgba(70, 201, 168, 0.9)";
  ctx.lineWidth = 3;
  ctx.fillStyle = "rgba(247, 248, 244, 0.95)";

  for (const [fromIndex, toIndex] of HAND_CONNECTIONS) {
    const from = landmarks[fromIndex];
    const to = landmarks[toIndex];
    ctx.beginPath();
    ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
    ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
    ctx.stroke();
  }

  landmarks.forEach((point, index) => {
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, index === 0 ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function syncCanvasToVideo() {
  if (
    canvas.width !== video.videoWidth ||
    canvas.height !== video.videoHeight
  ) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function resizeCanvas() {
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}

function logEvent(message) {
  const item = document.createElement("li");
  item.className = "event-item";
  item.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  eventLogEl.prepend(item);

  while (eventLogEl.children.length > 8) {
    eventLogEl.removeChild(eventLogEl.lastChild);
  }
}

function updateStatus(message) {
  statusPillEl.textContent = message;
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function cosineSimilarity(vectorA, vectorB) {
  if (!vectorA || !vectorB || vectorA.length !== vectorB.length) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let index = 0; index < vectorA.length; index += 1) {
    dot += vectorA[index] * vectorB[index];
    normA += vectorA[index] * vectorA[index];
    normB += vectorB[index] * vectorB[index];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / Math.sqrt(normA * normB);
}

function signWithDeadband(value, epsilon) {
  if (value > epsilon) {
    return 1;
  }
  if (value < -epsilon) {
    return -1;
  }
  return 0;
}

function resetSession() {
  gestureStabilizer.reset();
  opennessTracker.reset();
  repetitionEngine.reset();
  appState.repCount = 0;
  appState.phase = repetitionEngine.phaseLabel();
  appState.stableGesture = "-";
  appState.gestureScore = 0;
  appState.cosineSimilarity = 0;
  appState.openness = 0;
  appState.normalizedOpenness = 0;
  renderUI({
    openness: 0,
    normalized: 0,
    phaseLabel: repetitionEngine.phaseLabel(),
    similarity: 0
  });
  eventLogEl.innerHTML = "";
  logEvent("已重置计数和基线。");
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    updateStatus("启动失败");
    logEvent("摄像头或模型初始化失败，请检查权限。");
    startButton.disabled = false;
    startButton.textContent = "重新启动";
  }
});

resetButton.addEventListener("click", resetSession);

window.addEventListener("resize", syncCanvasToVideo);
window.addEventListener("beforeunload", () => {
  cancelAnimationFrame(rafId);
  webcamStream?.getTracks().forEach((track) => track.stop());
});

renderUI({
  openness: 0,
  normalized: 0,
  phaseLabel: repetitionEngine.phaseLabel(),
  similarity: 0
});
