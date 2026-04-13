/**
 * HassetsuViewer.tsx
 *
 * 射法八節の3Dアニメーションビューア
 * - Three.js + GLTFLoader でリグ付き人体モデル（Mixamo）を読み込み
 * - 各段階のポーズをスムーズ補間で再現
 * - 使用筋肉・骨格をカラーハイライト表示
 * - マウス/タッチによる回転・ズーム操作
 */

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  HASSETSU_STEPS,
  MUSCLE_MESH_KEYWORDS,
  BONE_ROTATION_MAP,
  type PoseAngles,
  type MuscleInfo,
} from "../data/hassetsuData";

/* ================================================================
   Types
   ================================================================ */
interface OrigMaterialInfo {
  color: THREE.Color | null;
  opacity: number;
  transparent: boolean;
  emissive: THREE.Color | null;
  emissiveIntensity: number;
}

/* ================================================================
   Component
   ================================================================ */
const HassetsuViewer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rootRef = useRef<THREE.Group | null>(null);
  const glbModelRef = useRef<THREE.Group | null>(null);
  const glbBonesRef = useRef<Record<string, THREE.Bone>>({});
  const origMaterialsRef = useRef<Map<THREE.Mesh, OrigMaterialInfo>>(new Map());
  const currentPoseRef = useRef<Record<string, number>>({});
  const targetPoseRef = useRef<Record<string, number>>({});
  const rafRef = useRef<number>(0);
  const tRef = useRef<number>(0);

  // Mouse/touch orbit
  const rotYRef = useRef(0.25);
  const rotXRef = useRef(0.04);
  const dragRef = useRef(false);
  const prevXRef = useRef(0);
  const prevYRef = useRef(0);

  const [currentStep, setCurrentStep] = useState(0);
  const [selectedMuscle, setSelectedMuscle] = useState<string | null>(null);
  const [selectedBtag, setSelectedBtag] = useState<number | null>(null);
  const [showLayers, setShowLayers] = useState({ sk: true, mu: true, jo: true });
  const [loadingStatus, setLoadingStatus] = useState<string | null>("GLBを読み込み中...");
  const [autoPlay, setAutoPlay] = useState(false);

  const step = HASSETSU_STEPS[currentStep];

  /* ── helpers ────────────────────────────────────── */
  const saveMeshOriginal = useCallback((mesh: THREE.Mesh) => {
    if (origMaterialsRef.current.has(mesh)) return;
    const mat = mesh.material as THREE.MeshStandardMaterial;
    if (!mat) return;
    origMaterialsRef.current.set(mesh, {
      color: mat.color ? mat.color.clone() : null,
      opacity: mat.opacity ?? 1,
      transparent: mat.transparent || false,
      emissive: mat.emissive ? mat.emissive.clone() : null,
      emissiveIntensity: mat.emissiveIntensity || 0,
    });
  }, []);

  const restoreMesh = useCallback((mesh: THREE.Mesh, orig: OrigMaterialInfo | undefined) => {
    if (!orig) return;
    const mat = mesh.material as THREE.MeshStandardMaterial;
    if (!mat) return;
    if (orig.color && mat.color) mat.color.copy(orig.color);
    mat.opacity = orig.opacity;
    mat.transparent = orig.transparent;
    if (orig.emissive && mat.emissive) mat.emissive.copy(orig.emissive);
    mat.emissiveIntensity = orig.emissiveIntensity;
    mat.needsUpdate = true;
  }, []);

  const isMuscleMatch = useCallback((meshName: string, targetMuscleName: string): boolean => {
    const def = MUSCLE_MESH_KEYWORDS[targetMuscleName];
    if (!def) return meshName.toLowerCase().includes(targetMuscleName.toLowerCase());
    const n = meshName.toLowerCase();
    return def.keys.some((k) => n.includes(k.toLowerCase()));
  }, []);

  /* ── GLB highlight ─────────────────────────────── */
  const applyGLBHighlight = useCallback(
    (targetName: string | null, mode: "muscle" | "bone") => {
      const model = glbModelRef.current;
      if (!model) return;

      model.traverse((obj) => {
        if (!(obj as THREE.Mesh).isMesh) return;
        const mesh = obj as THREE.Mesh;
        const mat = mesh.material as THREE.MeshStandardMaterial;
        if (!mat) return;
        saveMeshOriginal(mesh);

        if (mode === "muscle") {
          if (!targetName) {
            restoreMesh(mesh, origMaterialsRef.current.get(mesh));
          } else {
            const isMatch = isMuscleMatch(mesh.name, targetName);
            if (isMatch) {
              const color = MUSCLE_MESH_KEYWORDS[targetName]?.color || 0xff8844;
              if (mat.color) mat.color.setHex(color);
              mat.transparent = true;
              mat.opacity = 1.0;
              if (mat.emissive) {
                mat.emissive.setHex(color);
                mat.emissiveIntensity = 1.5;
              }
            } else {
              const layer = (mesh.userData as Record<string, string>).layer || "";
              if (layer === "muscle") {
                if (mat.color) mat.color.setRGB(0.1, 0.1, 0.1);
                mat.transparent = true;
                mat.opacity = 0.05;
                if (mat.emissive) mat.emissive.set(0x000000);
                mat.emissiveIntensity = 0;
              }
            }
            mat.needsUpdate = true;
          }
        } else if (mode === "bone") {
          if (targetName === "__bone_flash__") {
            const layer = (mesh.userData as Record<string, string>).layer || "";
            if (layer === "skeleton" || layer === "") {
              if (mat.emissive) mat.emissive.setHex(0x4a9fd4);
              mat.emissiveIntensity = 0.7;
              mat.needsUpdate = true;
            }
          } else {
            restoreMesh(mesh, origMaterialsRef.current.get(mesh));
          }
        }
      });
    },
    [saveMeshOriginal, restoreMesh, isMuscleMatch]
  );

  /* ── Layer visibility ──────────────────────────── */
  const updateGLBLayers = useCallback(
    (layers: { sk: boolean; mu: boolean; jo: boolean }) => {
      const model = glbModelRef.current;
      if (!model) return;
      model.traverse((obj) => {
        if (!(obj as THREE.Mesh).isMesh) return;
        const mesh = obj as THREE.Mesh;
        const layer = (mesh.userData as Record<string, string>).layer;
        if (layer === "skeleton") mesh.visible = layers.sk;
        else if (layer === "muscle") mesh.visible = layers.mu;
        else mesh.visible = true;
      });
    },
    []
  );

  /* ── Apply pose to GLB bones ───────────────────── */
  const applyPoseToGLB = useCallback((pose: Record<string, number>) => {
    const bones = glbBonesRef.current;
    for (const [boneName, fn] of Object.entries(BONE_ROTATION_MAP)) {
      const bone = bones[boneName];
      if (!bone) continue;
      const poseAngles: PoseAngles = {
        torsoX: pose.torsoX ?? 0, torsoZ: pose.torsoZ ?? 0,
        headX: pose.headX ?? 0, headY: pose.headY ?? 0,
        lShX: pose.lShX ?? 0, lShZ: pose.lShZ ?? 0,
        lElX: pose.lElX ?? 0, lElY: pose.lElY ?? 0, lElZ: pose.lElZ ?? 0,
        rShX: pose.rShX ?? 0, rShY: pose.rShY ?? 0, rShZ: pose.rShZ ?? 0,
        rElX: pose.rElX ?? 0, rElY: pose.rElY ?? 0, rElZ: pose.rElZ ?? 0,
        lScapX: pose.lScapX ?? 0, lScapY: pose.lScapY ?? 0,
        rScapX: pose.rScapX ?? 0, rScapY: pose.rScapY ?? 0,
        lHipX: pose.lHipX ?? 0, lHipZ: pose.lHipZ ?? 0,
        lFootY: pose.lFootY ?? 0, lKnX: pose.lKnX ?? 0,
        rHipX: pose.rHipX ?? 0, rHipZ: pose.rHipZ ?? 0,
        rFootY: pose.rFootY ?? 0, rKnX: pose.rKnX ?? 0,
      };
      const [x, y, z] = fn(poseAngles);
      bone.rotation.set(x, y, z);
    }
  }, []);

  /* ── Go to step ────────────────────────────────── */
  const goToStep = useCallback(
    (i: number) => {
      setCurrentStep(i);
      setSelectedMuscle(null);
      setSelectedBtag(null);

      const poseData = HASSETSU_STEPS[i].pose;
      const target: Record<string, number> = {};
      for (const [k, v] of Object.entries(poseData)) {
        target[k] = v as number;
      }
      targetPoseRef.current = target;

      // On first call, skip interpolation
      if (Object.keys(currentPoseRef.current).length === 0) {
        currentPoseRef.current = { ...target };
      }

      origMaterialsRef.current.clear();
      applyGLBHighlight(null, "muscle");
    },
    [applyGLBHighlight]
  );

  /* ── Muscle selection ──────────────────────────── */
  const handleSelectMuscle = useCallback(
    (name: string) => {
      if (selectedMuscle === name) {
        setSelectedMuscle(null);
        applyGLBHighlight(null, "muscle");
        return;
      }
      setSelectedMuscle(name);
      setSelectedBtag(null);
      applyGLBHighlight(name, "muscle");
    },
    [selectedMuscle, applyGLBHighlight]
  );

  /* ── Bone tag selection ────────────────────────── */
  const handleSelectBtag = useCallback(
    (idx: number) => {
      if (selectedBtag === idx) {
        setSelectedBtag(null);
        applyGLBHighlight(null, "bone");
        return;
      }
      setSelectedBtag(idx);
      setSelectedMuscle(null);
      applyGLBHighlight(null, "muscle");
      applyGLBHighlight("__bone_flash__", "bone");
      setTimeout(() => applyGLBHighlight(null, "bone"), 1200);
    },
    [selectedBtag, applyGLBHighlight]
  );

  /* ── Layer toggle ──────────────────────────────── */
  const toggleLayer = useCallback(
    (key: "sk" | "mu" | "jo") => {
      setShowLayers((prev) => {
        const next = { ...prev, [key]: !prev[key] };
        updateGLBLayers(next);
        return next;
      });
    },
    [updateGLBLayers]
  );

  /* ── Auto-play ─────────────────────────────────── */
  const autoPlayRef = useRef(autoPlay);
  const currentStepRef = useRef(currentStep);
  useEffect(() => { autoPlayRef.current = autoPlay; }, [autoPlay]);
  useEffect(() => { currentStepRef.current = currentStep; }, [currentStep]);

  useEffect(() => {
    if (!autoPlay) return;
    const interval = setInterval(() => {
      const next = (currentStepRef.current + 1) % HASSETSU_STEPS.length;
      goToStep(next);
    }, 3000);
    return () => clearInterval(interval);
  }, [autoPlay, goToStep]);

  /* ══════════════════════════════════════════════════
     Three.js initialization & render loop
     ══════════════════════════════════════════════════ */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x13202e);
    scene.fog = new THREE.Fog(0x13202e, 9, 20);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
    camera.position.set(0, 0.95, 3.3);
    camera.lookAt(0, 0.85, 0);
    cameraRef.current = camera;

    // Lights
    scene.add(new THREE.AmbientLight(0x8899cc, 0.7));
    const keyLight = new THREE.DirectionalLight(0xfff5e8, 1.2);
    keyLight.position.set(3, 5, 4);
    keyLight.castShadow = true;
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x4466aa, 0.45);
    fillLight.position.set(-3, 2, -2);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0x6688ff, 0.22);
    rimLight.position.set(0, 2, -4);
    scene.add(rimLight);

    // Floor
    const floorGeom = new THREE.CircleGeometry(1.8, 48);
    const floorMat = new THREE.MeshLambertMaterial({ color: 0x1a2d40 });
    const floor = new THREE.Mesh(floorGeom, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    scene.add(new THREE.GridHelper(3.6, 18, 0x2a4060, 0x1e3050));

    // Root group
    const root = new THREE.Group();
    scene.add(root);
    rootRef.current = root;

    // Resize
    const resize = () => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight || 520;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    setTimeout(resize, 30);

    // Mouse & touch controls
    const onMouseDown = (e: MouseEvent) => {
      dragRef.current = true;
      prevXRef.current = e.clientX;
      prevYRef.current = e.clientY;
    };
    const onTouchStart = (e: TouchEvent) => {
      dragRef.current = true;
      prevXRef.current = e.touches[0].clientX;
      prevYRef.current = e.touches[0].clientY;
    };
    const onMouseUp = () => { dragRef.current = false; };
    const onMouseMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      rotYRef.current += (e.clientX - prevXRef.current) * 0.012;
      rotXRef.current += (e.clientY - prevYRef.current) * 0.005;
      rotXRef.current = Math.max(-0.55, Math.min(0.75, rotXRef.current));
      prevXRef.current = e.clientX;
      prevYRef.current = e.clientY;
    };
    const onTouchMove = (e: TouchEvent) => {
      if (!dragRef.current) return;
      rotYRef.current += (e.touches[0].clientX - prevXRef.current) * 0.012;
      prevXRef.current = e.touches[0].clientX;
    };
    const onWheel = (e: WheelEvent) => {
      camera.position.z = Math.max(1.8, Math.min(6, camera.position.z + e.deltaY * 0.005));
    };

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("mouseup", onMouseUp);
    window.addEventListener("touchend", onMouseUp);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    canvas.addEventListener("wheel", onWheel, { passive: true });

    // ── Load GLB ──────────────────────────────────
    const loader = new GLTFLoader();
    loader.load(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (import.meta as any).env.BASE_URL + "human_rigged.glb",
      (gltf) => {
        const model = gltf.scene;

        // Scale & position
        const b0 = new THREE.Box3().setFromObject(model);
        const size0 = b0.getSize(new THREE.Vector3());
        const scale = 1.7 / size0.y;
        model.scale.setScalar(scale);
        model.updateMatrixWorld(true);
        const b1 = new THREE.Box3().setFromObject(model);
        const c1 = b1.getCenter(new THREE.Vector3());
        model.position.set(-c1.x, -b1.min.y, -c1.z);

        // Collect bones
        const bones: Record<string, THREE.Bone> = {};
        gltf.scene.traverse((obj) => {
          if ((obj as THREE.Bone).isBone && BONE_ROTATION_MAP[obj.name]) {
            bones[obj.name] = obj as THREE.Bone;
          }
        });
        // Fallback: SkinnedMesh.skeleton.bones
        if (Object.keys(bones).length === 0) {
          gltf.scene.traverse((obj) => {
            const skinned = obj as THREE.SkinnedMesh;
            if (skinned.isSkinnedMesh && skinned.skeleton) {
              skinned.skeleton.bones.forEach((b) => {
                if (BONE_ROTATION_MAP[b.name]) bones[b.name] = b;
              });
            }
          });
        }
        glbBonesRef.current = bones;

        // Materials
        model.traverse((obj) => {
          if (!(obj as THREE.Mesh).isMesh) return;
          const mesh = obj as THREE.Mesh;
          const userData = mesh.userData as Record<string, string>;
          if (!userData.layer) {
            if (mesh.name.startsWith("SK_")) userData.layer = "skeleton";
            else if (mesh.name.startsWith("MU_")) userData.layer = "muscle";
          }
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          mats.forEach((m) => {
            const mat = m as THREE.MeshStandardMaterial;
            if (mat && !mat.emissive) {
              mat.emissive = new THREE.Color(0);
              mat.emissiveIntensity = 0;
            }
          });
        });

        glbModelRef.current = model;
        root.add(model);
        origMaterialsRef.current.clear();

        // Apply initial pose
        const poseData = HASSETSU_STEPS[0].pose;
        const initial: Record<string, number> = {};
        for (const [k, v] of Object.entries(poseData)) {
          initial[k] = v as number;
        }
        currentPoseRef.current = { ...initial };
        targetPoseRef.current = { ...initial };
        applyPoseToGLB(initial);

        setLoadingStatus(null);
      },
      (xhr) => {
        if (xhr.total) {
          setLoadingStatus(`読み込み中... ${Math.round((xhr.loaded / xhr.total) * 100)}%`);
        }
      },
      (err) => {
        console.error("GLBエラー:", err);
        setLoadingStatus("GLBの読み込みに失敗しました");
      }
    );

    // ── Render loop ───────────────────────────────
    const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

    const loop = () => {
      rafRef.current = requestAnimationFrame(loop);

      // Pose interpolation
      const target = targetPoseRef.current;
      const current = currentPoseRef.current;
      for (const k in target) {
        if (typeof target[k] === "number") {
          current[k] = lerp(current[k] ?? 0, target[k], 0.08);
        } else {
          current[k] = target[k];
        }
      }
      if (glbModelRef.current) applyPoseToGLB(current);

      tRef.current += 0.014;
      const t = tRef.current;

      // Pulse animation for selected muscle
      if (glbModelRef.current) {
        const pulse = 0.4 + Math.sin(t * 3.5) * 0.15;
        glbModelRef.current.traverse((obj) => {
          if (!(obj as THREE.Mesh).isMesh) return;
          const mat = (obj as THREE.Mesh).material as THREE.MeshStandardMaterial;
          if (mat && mat.emissiveIntensity > 0.5) {
            mat.emissiveIntensity = pulse;
          }
        });
      }

      // Orbit
      root.rotation.y = rotYRef.current + Math.sin(t * 0.35) * 0.005;
      root.rotation.x = rotXRef.current;

      renderer.render(scene, camera);
    };
    rafRef.current = requestAnimationFrame(loop);

    // Cleanup
    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("mouseup", onMouseUp);
      window.removeEventListener("touchend", onMouseUp);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("wheel", onWheel);
      renderer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ══════════════════════════════════════════════════
     Render
     ══════════════════════════════════════════════════ */
  return (
    <div className="hassetsu-wrap">
      {/* ── Step buttons ── */}
      <div className="hassetsu-topbar">
        {HASSETSU_STEPS.map((s, i) => (
          <button
            key={i}
            className={`hassetsu-sbtn${currentStep === i ? " on" : ""}`}
            onClick={() => { setAutoPlay(false); goToStep(i); }}
          >
            {s.label}{s.name}
          </button>
        ))}
      </div>

      {/* ── 3D Canvas ── */}
      <div className="hassetsu-canvas-box">
        <canvas ref={canvasRef} id="hassetsu-canvas" />

        {/* Overlay buttons */}
        <div className="hassetsu-overlay">
          <button
            className={`hassetsu-lbtn${showLayers.sk ? " on" : ""}`}
            onClick={() => toggleLayer("sk")}
          >
            骨格
          </button>
          <button
            className={`hassetsu-lbtn${showLayers.mu ? " on" : ""}`}
            onClick={() => toggleLayer("mu")}
          >
            筋肉活性
          </button>
          <button
            className={`hassetsu-lbtn${showLayers.jo ? " on" : ""}`}
            onClick={() => toggleLayer("jo")}
          >
            関節
          </button>
        </div>

        {/* Auto-play button */}
        <div className="hassetsu-autoplay">
          <button
            className={`hassetsu-lbtn${autoPlay ? " on" : ""}`}
            onClick={() => setAutoPlay((p) => !p)}
          >
            {autoPlay ? "⏸ 停止" : "▶ 自動再生"}
          </button>
        </div>

        <div className="hassetsu-hint-txt">ドラッグ回転 / スクロールズーム</div>

        {/* Loading status */}
        {loadingStatus && (
          <div className="hassetsu-status">{loadingStatus}</div>
        )}
      </div>

      {/* ── Info panel ── */}
      <div className="hassetsu-bottom">
        <div className="hassetsu-pname">{step.name}</div>
        <div className="hassetsu-pdesc">{step.desc}</div>

        {/* Muscles */}
        <div className="hassetsu-sec">
          主要筋肉（明色=主働 / 淡色=協働）
        </div>
        <div className="hassetsu-chips">
          {step.muscles.map((m: MuscleInfo) => (
            <div
              key={m.name}
              className={`hassetsu-chip${selectedMuscle === m.name ? " sel" : ""}`}
              onClick={() => handleSelectMuscle(m.name)}
              title={`${m.nameEn}: ${m.function}`}
            >
              <div
                className="hassetsu-cdot"
                style={{ background: m.color }}
              />
              <span>{m.name}</span>
              <span className="hassetsu-rl">{m.role}働</span>
            </div>
          ))}
        </div>

        {/* Muscle detail */}
        {selectedMuscle && (
          <div className="hassetsu-muscle-detail">
            {step.muscles
              .filter((m) => m.name === selectedMuscle)
              .map((m) => (
                <div key={m.name} className="hassetsu-muscle-info">
                  <div className="hassetsu-muscle-name-en">{m.nameEn}</div>
                  <div className="hassetsu-muscle-func">{m.function}</div>
                </div>
              ))}
          </div>
        )}

        {/* Anatomy tags */}
        <div className="hassetsu-sec" style={{ marginTop: 7 }}>
          解剖学的ポイント
        </div>
        <div className="hassetsu-blist">
          {step.anatomyTags.map((tag, i) => (
            <div
              key={i}
              className={`hassetsu-btag${selectedBtag === i ? " sel" : ""}`}
              onClick={() => handleSelectBtag(i)}
            >
              📌 {tag}
            </div>
          ))}
        </div>

        {/* Reference */}
        <div className="hassetsu-ref">参考：{step.ref}</div>

        {selectedMuscle && (
          <div className="hassetsu-hint-sel">
            ↑ クリックでフォーカス解除 / 別の筋肉をクリックで切替
          </div>
        )}
      </div>
    </div>
  );
};

export default HassetsuViewer;
