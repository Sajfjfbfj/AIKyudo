import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { POSE_CONNECTIONS } from '@mediapipe/pose';

// ── MediaPipe Pose ランドマーク番号 ───────────────────────────────
// 0:  鼻          1-6: 目(内/中/外)    7: 左耳    8: 右耳
// 11: 左肩        12: 右肩
// 13: 左肘        14: 右肘
// 15: 左手首      16: 右手首
// 23: 左腰        24: 右腰
// ─────────────────────────────────────────────────────────────────

export interface AngleData {
  leftElbow:     number | null;  // 左肘角度
  rightElbow:    number | null;  // 右肘角度
  leftShoulder:  number | null;  // 左肩角度
  rightShoulder: number | null;  // 右肩角度
  hipTilt:       number | null;  // 腰ライン傾き(°)
  spineTilt:     number | null;  // 背筋傾き(°)
  monomiAngle:   number | null;  // 物見角度(°) — 頭の回転量
  kuchiwariOffset: number | null; // 口割オフセット — 右手首が口より高い(+)/低い(-) 正規化座標差
}

// ── 骨格描画 ─────────────────────────────────────────────────────
export function drawPoseOverlay(
  ctx: CanvasRenderingContext2D,
  landmarks: any[],
  canvasWidth: number,
  canvasHeight: number
): void {
  // 骨格ライン
  drawConnectors(ctx, landmarks, POSE_CONNECTIONS, {
    color: 'rgba(0, 255, 180, 0.85)',
    lineWidth: 2,
  });
  // ランドマーク点
  drawLandmarks(ctx, landmarks, {
    color: '#FF4060',
    lineWidth: 1,
    radius: 4,
  });

  // 重要な関節をハイライト
  const highlighted = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
  for (const idx of highlighted) {
    const lm = landmarks[idx];
    if (!lm) continue;
    const x = lm.x * canvasWidth;
    const y = lm.y * canvasHeight;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 220, 50, 0.88)';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // ── 物見ライン（耳のライン）を水色で描画 ──────────────────────
  const lEar = landmarks[7];
  const rEar = landmarks[8];
  if (lEar && rEar) {
    ctx.beginPath();
    ctx.moveTo(lEar.x * canvasWidth, lEar.y * canvasHeight);
    ctx.lineTo(rEar.x * canvasWidth, rEar.y * canvasHeight);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    // ラベル
    ctx.fillStyle = '#38bdf8';
    ctx.font = `bold 11px 'IBM Plex Mono', monospace`;
    ctx.fillText('物見', rEar.x * canvasWidth + 6, rEar.y * canvasHeight - 4);
  }

  // ── 口割ライン（右手首の高さを水平線で可視化）─────────────────
  const rWrist = landmarks[16];
  const nose   = landmarks[0];
  if (rWrist && nose) {
    // 口の推定位置（鼻 + 顎方向へオフセット）
    const mouthY = (nose.y + (nose.y + 0.06)) / 2; // 鼻より約3%下を口と推定

    // 右手首の高さを示す水平線
    const wristY = rWrist.y * canvasHeight;
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.moveTo(0, wristY);
    ctx.lineTo(canvasWidth, wristY);
    ctx.strokeStyle = rWrist.y < mouthY + 0.02 ? '#f97316' : '#a3e635';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f97316';
    ctx.font = `bold 11px 'IBM Plex Mono', monospace`;
    ctx.fillText('口割', rWrist.x * canvasWidth + 8, wristY - 4);
  }
}

// ── 3点の角度計算（頂点 b）────────────────────────────────────────
export function calcAngle(a: any, b: any, c: any): number {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.sqrt(ab.x ** 2 + ab.y ** 2) * Math.sqrt(cb.x ** 2 + cb.y ** 2);
  if (mag === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
}

// ── 弓道全関節角度 + 物見 + 口割 の計算 ──────────────────────────
export function calcKyudoAngles(landmarks: any[]): AngleData {
  const nose      = landmarks[0];
  const lEar      = landmarks[7];
  const rEar      = landmarks[8];
  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const lElbow    = landmarks[13];
  const rElbow    = landmarks[14];
  const lWrist    = landmarks[15];
  const rWrist    = landmarks[16];
  const lHip      = landmarks[23];
  const rHip      = landmarks[24];

  // ── 既存の角度 ────────────────────────────────────────────────
  const leftElbow =
    lShoulder && lElbow && lWrist ? calcAngle(lShoulder, lElbow, lWrist) : null;
  const rightElbow =
    rShoulder && rElbow && rWrist ? calcAngle(rShoulder, rElbow, rWrist) : null;
  const leftShoulder =
    lElbow && lShoulder && rShoulder ? calcAngle(lElbow, lShoulder, rShoulder) : null;
  const rightShoulder =
    lShoulder && rShoulder && rElbow ? calcAngle(lShoulder, rShoulder, rElbow) : null;

  // 腰ラインの傾き（°）
  let hipTilt: number | null = null;
  if (lHip && rHip) {
    hipTilt = Math.abs(
      Math.atan2(rHip.y - lHip.y, rHip.x - lHip.x) * (180 / Math.PI)
    );
  }

  // 背筋の傾き（°）
  let spineTilt: number | null = null;
  if (lShoulder && rShoulder && lHip && rHip) {
    const sMid = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
    const hMid = { x: (lHip.x + rHip.x) / 2,           y: (lHip.y + rHip.y) / 2 };
    spineTilt = Math.abs(
      Math.atan2(sMid.x - hMid.x, hMid.y - sMid.y) * (180 / Math.PI)
    );
  }

  // ── 物見角度 ────────────────────────────────────────────────────
  //
  //  【計算方法】
  //  1. 肩ラインのベクトル: lShoulder → rShoulder
  //  2. 耳ラインのベクトル: lEar → rEar
  //  3. 二つのベクトルの成す角 = 頭の肩に対する回転量（物見角度）
  //
  //  弓道では標的方向（右側）へ頭を回すため、右耳が右肩より内側に入ると角度が増す。
  //  理想：45° 前後（弓道教本：顔を的に向け、頬付けができる位置）
  //
  let monomiAngle: number | null = null;
  if (lEar && rEar && lShoulder && rShoulder) {
    // 肩ラインの角度（水平からの傾き）
    const shoulderAngleRad = Math.atan2(
      rShoulder.y - lShoulder.y,
      rShoulder.x - lShoulder.x
    );
    // 耳ラインの角度
    const earAngleRad = Math.atan2(
      rEar.y - lEar.y,
      rEar.x - lEar.x
    );
    // 差分（耳ラインが肩ラインより右に回転している量）
    let diff = (earAngleRad - shoulderAngleRad) * (180 / Math.PI);
    // -180〜180 に正規化
    if (diff > 180)  diff -= 360;
    if (diff < -180) diff += 360;
    monomiAngle = diff; // 正 = 右向き、負 = 左向き
  }

  // ── 口割オフセット ──────────────────────────────────────────────
  //
  //  【計算方法】
  //  MediaPipe Pose に口の直接ランドマークはないため、鼻(0)を基準に推定。
  //  顔の縦幅（鼻 y ～ 耳の y）の比率から口の位置を概算。
  //
  //  kuchiwariOffset = mouthEstimatedY - rWrist.y  (正規化座標)
  //    +0.0 付近 → 正しい口割（右手首が口の高さ）
  //    + 大きい  → 口割が低すぎ（矢が下に飛びやすい）
  //    - 大きい  → 口割が高すぎ（頬付けが額付けになっている）
  //
  let kuchiwariOffset: number | null = null;
  if (nose && rWrist && lEar && rEar) {
    // 耳の平均 y を「あご付近の参照点」として使う（耳は鼻より低い位置にある）
    const earMidY = (lEar.y + rEar.y) / 2;
    // 口の推定 y = 鼻 y と 耳中点 y の中間よりやや下
    const estimatedMouthY = nose.y + (earMidY - nose.y) * 0.55;
    // 口割オフセット（右手首との差）
    // y は下ほど大きい値なので、 rWrist.y > mouthY → 手首が下 → 正の値
    kuchiwariOffset = rWrist.y - estimatedMouthY;
  }

  return {
    leftElbow,
    rightElbow,
    leftShoulder,
    rightShoulder,
    hipTilt,
    spineTilt,
    monomiAngle,
    kuchiwariOffset,
  };
}