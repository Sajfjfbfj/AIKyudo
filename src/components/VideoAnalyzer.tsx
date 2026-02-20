/**
 * VideoAnalyzer.tsx
 * リプレイ機能付き弓道フォーム解析
 *
 * 解析済みランドマークを storedLandmarks に保存し、
 * リプレイ時は MediaPipe を再実行せず描画のみを行う。
 * 速度変更（0.25x / 0.5x / 1x / 2x）・シークバー対応。
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Pose, Results } from '@mediapipe/pose';
import { drawPoseOverlay, calcKyudoAngles } from './PoseOverlay';
import AngleChart, { FrameAngleData } from './AngleChart';

interface VideoAnalyzerProps {
  videoSrc: string;
}

interface StoredFrame {
  frame:     number;
  timeMs:    number;   // video.currentTime × 1000
  landmarks: any[];
}

interface EvalItem {
  label:   string;
  score:   number;
  comment: string;
  detail:  string;
  ideal:   string;
}

interface FormEvaluation {
  total: number;
  rank:  string;
  items: EvalItem[];
}

// ── 統計ヘルパー ──────────────────────────────────────────────────
const mean   = (v: number[]) => v.length === 0 ? 0 : v.reduce((a,b) => a+b,0)/v.length;
const stddev = (v: number[]) => {
  if (v.length < 2) return 0;
  const m = mean(v);
  return Math.sqrt(v.reduce((a,b) => a+(b-m)**2, 0)/v.length);
};
const nn    = (v: (number|null)[]): number[] => v.filter((x): x is number => x !== null);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// ── フォーム評価（前回実装と同一） ─────────────────────────────────
function evaluateForm(frames: FrameAngleData[]): FormEvaluation {
  if (frames.length === 0) return { total:0, rank:'—', items:[] };
  const items: EvalItem[] = [];

  const leVals = nn(frames.map(f => f.leftElbow));
  if (leVals.length > 5) {
    const peak = Math.max(...leVals);
    const score =
      peak>=160&&peak<=172?100: peak>=150&&peak<160?60+(peak-150)*4:
      peak>172&&peak<=178?100-(peak-172)*10: peak>178?40: clamp((peak-120)*2,0,60);
    items.push({ label:'押し手（左肘）の骨法', score:Math.round(clamp(score,0,100)),
      comment: peak>=160&&peak<=172?'✅ 押し手の骨法が正しく出ています': peak>178?'❌ 押し手が伸びすぎています（弦打ちリスク）': peak>172?'⚠️ 押し手がやや伸びすぎです': peak>=150?'⚠️ 押し手の伸びがやや不足です':'❌ 押し手が大きく曲がっています',
      detail:'弓道教本：「肘を完全に伸ばしきると弓手肩が突っ張り、馬手肩が後ろに抜けやすくなる」', ideal:'会でのピーク角度 160〜172°' });
  }

  const reVals = nn(frames.map(f => f.rightElbow));
  if (reVals.length > 5) {
    const late=reVals.slice(Math.floor(reVals.length*0.5)), avg=mean(late);
    const score= avg>=80&&avg<=110?100: avg>110&&avg<=125?100-(avg-110)*3: avg>125?clamp(100-(avg-110)*5,0,55): clamp(80+(avg-70)*2,0,80);
    items.push({ label:'馬手（右肘）の収まり', score:Math.round(clamp(score,0,100)),
      comment: avg>=80&&avg<=110?'✅ 馬手肘が正しく収まっています': avg>125?'❌ 馬手の前収まり。緩み離れのリスクがあります': avg>110?'⚠️ 馬手肘がやや前収まりです':'⚠️ 引き分けを確認してください',
      detail:'理論弓道：「前収まりだとほぼ100%緩み離れになる」', ideal:'引き分け後半の肘角度 80〜110°' });
  }

  const lsVals=nn(frames.map(f=>f.leftShoulder)), rsVals=nn(frames.map(f=>f.rightShoulder));
  if (lsVals.length>5&&rsVals.length>5) {
    const diff=Math.abs(mean(lsVals)-mean(rsVals)), avgStab=(stddev(lsVals)+stddev(rsVals))/2;
    const score=clamp(100-diff*3,0,100)*0.6+clamp(100-avgStab*3,0,100)*0.4;
    items.push({ label:'引き分けの左右均等性', score:Math.round(clamp(score,0,100)),
      comment: diff<=8&&avgStab<=10?'✅ 左右均等な引き分けができています': diff<=15?'⚠️ わずかに左右差があります':'❌ 左右差が大きいです',
      detail:'弓道教本：「胸の中筋から左右に開くように体を弓の中に割って入る」', ideal:'左右肩角度差 8° 以内' });
  }

  const hipVals=nn(frames.map(f=>f.hipTilt));
  if (hipVals.length>5) {
    const score=clamp(100-mean(hipVals)*9,0,100)*0.65+clamp(100-stddev(hipVals)*6,0,100)*0.35;
    items.push({ label:'三重十文字（肩・腰ラインの水平性）', score:Math.round(clamp(score,0,100)),
      comment: mean(hipVals)<=4&&stddev(hipVals)<=4?'✅ 三重十文字が安定しています': mean(hipVals)<=7?'⚠️ わずかに傾きがあります': mean(hipVals)<=13?'❌ 傾きが目立ちます':'❌ 三重十文字が大きく崩れています',
      detail:'弓道用語辞典：「足底・腰・肩の線が上から見たときに一枚になる状態」', ideal:'腰ライン傾き 4° 以内' });
  }

  const spineVals=nn(frames.map(f=>f.spineTilt));
  if (spineVals.length>5) {
    const score=clamp(100-mean(spineVals)*7,0,100)*0.65+clamp(100-stddev(spineVals)*5,0,100)*0.35;
    items.push({ label:'胴造り（背筋の垂直性）', score:Math.round(clamp(score,0,100)),
      comment: mean(spineVals)<=4&&stddev(spineVals)<=5?'✅ 胴造りが正しく保たれています': mean(spineVals)<=8?'⚠️ やや「胴が入る・起きる」傾向があります': mean(spineVals)<=14?'❌ 胴の傾きが大きいです':'❌ 胴造りが大きく崩れています',
      detail:'弓道教本：「重心を総体の中心に置き、前後左右に傾かない垂直な軸を作る」', ideal:'脊柱傾き 4° 以内' });
  }

  const lateLE=leVals.slice(Math.floor(leVals.length*0.55)), lateRE=reVals.slice(Math.floor(reVals.length*0.55));
  if (lateLE.length>5&&lateRE.length>5) {
    const avgStd=(stddev(lateLE)+stddev(lateRE))/2, kaiSec=frames.length/30*0.33;
    const kaiBonus=kaiSec>=3?0:kaiSec>=1.5?-10:-25;
    const score=clamp(100-avgStd*5+kaiBonus,0,100);
    items.push({ label:'会の保持安定性（詰め合い・伸び合い）', score:Math.round(clamp(score,0,100)),
      comment: avgStd<=4&&kaiBonus===0?'✅ 会が充実しています': kaiBonus===-25?'❌ 早気の可能性があります（会を3秒以上保ちましょう）': avgStd<=9?'⚠️ 会中にブレがあります':'❌ 会が大きく不安定です',
      detail:'宇野範士（弓道教本）：「全力で十文字に伸び合う」。会3秒未満は早気の可能性（武道学研究）', ideal:'後半フレームの角度ブレ 4° 以内、推定会時間 3秒以上' });
  }

  if (leVals.length>10) {
    const deltas=leVals.slice(1).map((v,i)=>Math.abs(v-leVals[i]));
    const score=clamp(100-stddev(deltas)*18-Math.max(0,mean(deltas)-1.5)*10,0,100);
    items.push({ label:'引き分けの滑らかさ', score:Math.round(clamp(score,0,100)),
      comment: score>=82?'✅ 滑らかで均一な引き分けができています': score>=62?'⚠️ 引き分けにやや引っかかりがあります':'❌ つかみ引きや途中止まりがないか確認しましょう',
      detail:'弓道教本：「遅速なく左右均等に引き分ける」', ideal:'フレーム間角度変化の標準偏差が小さいこと' });
  }

  const monomiVals=nn(frames.map(f=>f.monomiAngle));
  if (monomiVals.length>5) {
    const late=monomiVals.slice(Math.floor(monomiVals.length*0.3)), avg=mean(late), std=stddev(late);
    if (!isNaN(avg) && !isNaN(std)) {
      const aScore= avg>=35&&avg<=55?100: avg>=25&&avg<35?60+(avg-25)*4: avg>55&&avg<=68?100-(avg-55)*4: avg<25?clamp(avg*2.4,0,60): clamp(100-(avg-55)*6,0,60);
      const score=aScore*0.65+clamp(100-std*5,0,100)*0.35;
      items.push({ label:'物見（頭の向き・安定性）', score:Math.round(clamp(score,0,100)),
        comment: avg>=35&&avg<=55&&std<=6?'✅ 物見が正しい角度で安定しています': avg<25?'❌ 物見が浅すぎます（照的になりやすい）': avg<35?'⚠️ 物見がやや浅いです': avg>68?'⚠️ 物見が深すぎます': std>10?'⚠️ 物見が引き分け中にブレています':'⚠️ 物見の角度を確認しましょう',
        detail:'弓道教本・弓道大学：「物見は45°程度が理想。浅い物見（照的）は矢所が定まらない原因」', ideal:'耳ライン・肩ライン角度差 35〜55°、ブレ 6° 以内' });
    }
  }

  const kVals=nn(frames.map(f=>f.kuchiwariOffset));
  if (kVals.length>5) {
    const late=kVals.slice(Math.floor(kVals.length*0.5)), avg=mean(late), std=stddev(late);
    if (!isNaN(avg) && !isNaN(std)) {
      const pScore= avg>=-0.01&&avg<=0.03?100: avg>0.03&&avg<=0.07?100-(avg-0.03)*1000: avg<-0.01&&avg>=-0.05?100+(avg+0.01)*1000: clamp(50-Math.abs(avg)*500,0,50);
      const score=pScore*0.7+clamp(100-std*1000,0,100)*0.3;
      items.push({ label:'口割（右手首の高さ）', score:Math.round(clamp(score,0,100)),
        comment: avg>=-0.01&&avg<=0.03&&std<=0.02?'✅ 口割が正しい位置に安定しています': avg>0.05?'❌ 口割が低すぎます（矢が上に飛びやすい）': avg>0.03?'⚠️ 口割がやや低いです': avg<-0.05?'❌ 口割が高すぎます（額付けの可能性）': avg<-0.01?'⚠️ 口割がやや高いです': std>0.02?'⚠️ 口割が引き分け中にブレています':'⚠️ 口割の位置を確認しましょう',
        detail:'弓道教本：「会における右手は口の高さに収まる。毎射一定に保つことで矢所が安定する」', ideal:'右手首が口の高さ ±2cm 程度（正規化座標差 -0.01〜+0.03）' });
    }
  }

  if (items.length===0) return { total:0, rank:'—', items:[] };
  const total=Math.round(items.reduce((a,b)=>a+b.score,0)/items.length);
  const rank= total>=90?'四〜五段相当': total>=78?'三段相当': total>=65?'二段相当': total>=52?'初段相当': total>=38?'級位相当':'要基礎練習';
  return { total, rank, items };
}

const scoreColor = (s: number) => s>=80?'#34d399': s>=55?'#fbbf24':'#f87171';

const SPEEDS = [0.25, 0.5, 1.0, 2.0];

// ════════════════════════════════════════════════════════════════
//  コンポーネント
// ════════════════════════════════════════════════════════════════
const VideoAnalyzer: React.FC<VideoAnalyzerProps> = ({ videoSrc }) => {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef   = useRef<Pose | null>(null);
  const rafRef    = useRef<number | null>(null);
  const frameRef  = useRef(0);
  const procRef   = useRef(false);

  // リプレイ用ランドマーク保存
  const storedFramesRef = useRef<StoredFrame[]>([]);

  const [frameAngles,  setFrameAngles]  = useState<FrameAngleData[]>([]);
  const [displayCount, setDisplayCount] = useState(0);
  const [status,       setStatus]       = useState<'loading'|'playing'|'done'|'error'>('loading');
  const [evaluation,   setEvaluation]   = useState<FormEvaluation | null>(null);
  const [expandedItem, setExpandedItem] = useState<string | null>(null);

  // ── リプレイ状態 ──────────────────────────────────────────────
  const [isReplaying,   setIsReplaying]   = useState(false);
  const [replayPaused,  setReplayPaused]  = useState(false);
  const [replaySpeed,   setReplaySpeed]   = useState(1.0);
  const [replayFrame,   setReplayFrame]   = useState(0);   // 現在のリプレイフレームindex
  const [totalFrames,   setTotalFrames]   = useState(0);
  const replayIdxRef    = useRef(0);
  const replayRafRef    = useRef<number | null>(null);
  const replayPausedRef = useRef(false);

  // ── ダウンロード ───────────────────────────────────────────────
  const handleDownload = () => {
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(new Blob([JSON.stringify({frames:frameAngles},null,2)], {type:'application/json'})),
      download: 'kyudo_analysis.json',
    });
    a.click();
  };

  // ── 解析ループ ─────────────────────────────────────────────────
  const startProcessing = useCallback(() => {
    const video=videoRef.current, canvas=canvasRef.current, pose=poseRef.current;
    if (!video||!canvas||!pose) return;

    const loop = async () => {
      if (!video||video.ended) { setStatus('done'); return; }
      if (video.paused)        { rafRef.current=requestAnimationFrame(loop); return; }
      if (procRef.current)     { rafRef.current=requestAnimationFrame(loop); return; }
      procRef.current=true;
      try { await pose.send({ image: video }); } catch {}
      procRef.current=false;
      rafRef.current=requestAnimationFrame(loop);
    };
    rafRef.current=requestAnimationFrame(loop);
  }, []);

  // ── リプレイループ（保存済みランドマークを再描画）───────────────
  const stopReplay = useCallback(() => {
    if (replayRafRef.current) { cancelAnimationFrame(replayRafRef.current); replayRafRef.current=null; }
    setIsReplaying(false);
    setReplayPaused(false);
    replayPausedRef.current=false;
  }, []);

  const startReplay = useCallback((startIdx = 0, speed = replaySpeed) => {
    const canvas=canvasRef.current;
    const video=videoRef.current;
    if (!canvas||!video||storedFramesRef.current.length===0) return;

    if (replayRafRef.current) cancelAnimationFrame(replayRafRef.current);

    replayIdxRef.current=startIdx;
    setIsReplaying(true);
    setReplayPaused(false);
    replayPausedRef.current=false;

    const stored=storedFramesRef.current;
    const ctx=canvas.getContext('2d');
    if (!ctx) return;

    // フレーム間の時間差を速度で割って待機
    let lastRealTime: number | null = null;
    let lastFrameTime = stored[startIdx]?.timeMs ?? 0;

    const loop = (now: number) => {
      if (replayPausedRef.current) { replayRafRef.current=requestAnimationFrame(loop); return; }

      const idx=replayIdxRef.current;
      if (idx >= stored.length) { stopReplay(); return; }

      // 前フレームからの経過時間（実時間）
      if (lastRealTime === null) lastRealTime=now;
      const realElapsed=(now-lastRealTime);
      lastRealTime=now;

      // 動画時間でどこまで進むか
      lastFrameTime += realElapsed * speed;

      // lastFrameTime に対応するフレームを探す
      let nextIdx=idx;
      while (nextIdx < stored.length-1 && stored[nextIdx].timeMs <= lastFrameTime) nextIdx++;

      replayIdxRef.current=nextIdx;
      setReplayFrame(nextIdx);

      const frame=stored[nextIdx];
      const targetTimeMs=frame.timeMs;
      video.currentTime = targetTimeMs / 1000;

      // canvas に描画
      ctx.clearRect(0,0,canvas.width,canvas.height);
      
      // 動画フレームを描画
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      } catch (e) {
        // ビデオフレームが利用できない場合は背景色を使用
        ctx.fillStyle='#0a0f1e';
        ctx.fillRect(0,0,canvas.width,canvas.height);
      }

      // フレーム番号表示
      ctx.fillStyle='rgba(255,255,255,0.15)';
      ctx.font=`12px 'IBM Plex Mono', monospace`;
      ctx.fillText(`FRAME ${frame.frame}`, 12, 20);

      // 骨格描画
      drawPoseOverlay(ctx, frame.landmarks, canvas.width, canvas.height);

      replayRafRef.current=requestAnimationFrame(loop);
    };

    replayRafRef.current=requestAnimationFrame(loop);
  }, [replaySpeed, stopReplay]);

  // リプレイ一時停止/再開
  const toggleReplayPause = () => {
    const next=!replayPausedRef.current;
    replayPausedRef.current=next;
    setReplayPaused(next);
  };

  // シークバー変更
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const idx=Number(e.target.value);
    replayIdxRef.current=idx;
    setReplayFrame(idx);

    // 一時停止中でも即座に描画更新
    const canvas=canvasRef.current;
    const video=videoRef.current;
    const stored=storedFramesRef.current;
    if (!canvas||!video||!stored[idx]) return;
    const ctx=canvas.getContext('2d');
    if (!ctx) return;

    const frame=stored[idx];
    const targetTimeMs=frame.timeMs;
    video.currentTime = targetTimeMs / 1000;

    ctx.clearRect(0,0,canvas.width,canvas.height);
    
    // 動画フレームを描画
    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } catch (e) {
      // ビデオフレームが利用できない場合は背景色を使用
      ctx.fillStyle='#0a0f1e';
      ctx.fillRect(0,0,canvas.width,canvas.height);
    }

    ctx.fillStyle='rgba(255,255,255,0.15)';
    ctx.font=`12px 'IBM Plex Mono', monospace`;
    ctx.fillText(`FRAME ${stored[idx].frame}`, 12, 20);
    drawPoseOverlay(ctx, stored[idx].landmarks, canvas.width, canvas.height);
  };

  // ── MediaPipe セットアップ ────────────────────────────────────
  useEffect(() => {
    const video=videoRef.current, canvas=canvasRef.current;
    if (!video||!canvas) return;

    frameRef.current=0; procRef.current=false;
    storedFramesRef.current=[];
    setFrameAngles([]); setDisplayCount(0); setEvaluation(null);
    setStatus('loading'); setIsReplaying(false); setReplayFrame(0); setTotalFrames(0);
    if (rafRef.current)    { cancelAnimationFrame(rafRef.current);    rafRef.current=null; }
    if (replayRafRef.current) { cancelAnimationFrame(replayRafRef.current); replayRafRef.current=null; }

    const pose=new Pose({ locateFile: f=>`https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}` });
    pose.setOptions({ modelComplexity:1, smoothLandmarks:true, enableSegmentation:false,
      minDetectionConfidence:0.5, minTrackingConfidence:0.5 });

    pose.onResults((results: Results) => {
      const ctx=canvas.getContext('2d');
      if (!ctx||!video) return;
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(video,0,0,canvas.width,canvas.height);

      if (results.poseLandmarks) {
        drawPoseOverlay(ctx, results.poseLandmarks, canvas.width, canvas.height);
        const angles=calcKyudoAngles(results.poseLandmarks);
        const current=frameRef.current++;

        // ランドマーク保存（リプレイ用）
        storedFramesRef.current.push({ frame:current, timeMs:video.currentTime*1000, landmarks:results.poseLandmarks });

        setFrameAngles(p=>[...p, { frame:current, ...angles }]);
        setDisplayCount(frameRef.current);
      }
    });

    poseRef.current=pose;

    const onMeta=()=>{
      canvas.width=video.videoWidth||640;
      canvas.height=video.videoHeight||360;
      video.play()
        .then(()=>{ setStatus('playing'); startProcessing(); })
        .catch(()=>setStatus('error'));
    };
    video.addEventListener('loadedmetadata', onMeta);
    video.addEventListener('ended', ()=>setStatus('done'));
    video.addEventListener('error', ()=>setStatus('error'));
    video.src=videoSrc; video.load();

    return ()=>{
      if (rafRef.current)       { cancelAnimationFrame(rafRef.current);       rafRef.current=null; }
      if (replayRafRef.current) { cancelAnimationFrame(replayRafRef.current); replayRafRef.current=null; }
      video.pause(); pose.close();
    };
  }, [videoSrc, startProcessing]);

  useEffect(()=>{
    if (status==='done'&&frameAngles.length>0) {
      setEvaluation(evaluateForm(frameAngles));
      setTotalFrames(storedFramesRef.current.length);
    }
  }, [status, frameAngles]);

  return (
    <div className="analyzer-wrap">
      <video ref={videoRef} crossOrigin="anonymous" playsInline muted
        style={{ position:'absolute', width:1, height:1, opacity:0, pointerEvents:'none' }} />

      {/* ロード中・エラー時の表示 */}
      {status==='loading' && (
        <div className="status-overlay loading">
          <div className="status-content">
            <div className="spinner"></div>
            <p>⏳ モデル読み込み中...</p>
          </div>
        </div>
      )}
      
      {status==='error' && (
        <div className="status-overlay error">
          <div className="status-content">
            <p>❌ 動画の読み込みに失敗しました</p>
          </div>
        </div>
      )}

      {status==='playing' && (
        <div className="status-overlay loading">
          <div className="status-content">
            <div className="spinner"></div>
            <p>🔍 動画を解析中...</p>
          </div>
        </div>
      )}

      {/* キャンバス（解析完了後のみ表示） */}
      <canvas ref={canvasRef} className="pose-canvas"
        style={{ display: status==='done'?'block':'none' }} />


      {/* ══ リプレイコントロール（解析完了後のみ） ══ */}
      {status==='done' && totalFrames>0 && (
        <div className="replay-panel">
          <div className="replay-header">
            <span className="replay-title">🎬 リプレイ</span>
            <span className="replay-framecount">{isReplaying ? replayFrame : displayCount} / {totalFrames} フレーム</span>
          </div>

          {/* シークバー */}
          <input
            type="range"
            className="replay-seekbar"
            min={0}
            max={totalFrames - 1}
            value={isReplaying ? replayFrame : 0}
            onChange={handleSeek}
            onMouseDown={()=>{ if (!isReplaying) { startReplay(0); replayPausedRef.current=true; setReplayPaused(true); } }}
          />

          {/* ボタン群 */}
          <div className="replay-controls">
            {/* 再生/停止 */}
            {!isReplaying ? (
              <button className="replay-btn primary" onClick={()=>startReplay(0, replaySpeed)}>
                ▶ 再生
              </button>
            ) : (
              <>
                <button className="replay-btn" onClick={toggleReplayPause}>
                  {replayPaused ? '▶ 再開' : '⏸ 一時停止'}
                </button>
                <button className="replay-btn" onClick={stopReplay}>
                  ⏹ 停止
                </button>
              </>
            )}

            {/* 速度変更 */}
            <div className="replay-speed-group">
              {SPEEDS.map(s=>(
                <button
                  key={s}
                  className={`replay-speed-btn${replaySpeed===s?' active':''}`}
                  onClick={()=>{
                    setReplaySpeed(s);
                    if (isReplaying) startReplay(replayIdxRef.current, s);
                  }}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {frameAngles.length>10 && <AngleChart data={frameAngles} />}

      {evaluation && (
        <div className="eval-section">
          <h3 className="eval-title">AI フォーム評価</h3>
          <p className="eval-disclaimer">
            ※ AI診断なので間違う場合がございます。参考値として御覧ください。
          </p>
          <div className="eval-items">
            {evaluation.items.map(item=>(
              <div className="eval-item" key={item.label}>
                <div className="eval-item-header"
                  onClick={()=>setExpandedItem(expandedItem===item.label?null:item.label)}
                  style={{cursor:'pointer'}}>
                  <span className="eval-item-label">{item.label}</span>
                  <div style={{display:'flex',alignItems:'center',gap:8}}>
                    <span className="eval-item-score" style={{color:scoreColor(item.score)}}>{item.score}点</span>
                    <span className="eval-expand-icon">{expandedItem===item.label?'▲':'▼'}</span>
                  </div>
                </div>
                <div className="eval-bar-bg">
                  <div className="eval-bar-fill" style={{width:`${item.score}%`,background:scoreColor(item.score)}} />
                </div>
                <p className="eval-comment">{item.comment}</p>
                {expandedItem===item.label&&(
                  <div className="eval-detail">
                    <p className="eval-detail-basis"><strong>弓道的根拠：</strong>{item.detail}</p>
                    <p className="eval-detail-ideal"><strong>理想値：</strong>{item.ideal}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
          <button className="download-btn" onClick={handleDownload}>📥 解析データを JSON でダウンロード</button>
        </div>
      )}
    </div>
  );
};

export default VideoAnalyzer;