/**
 * ModelViewer.tsx
 * Three.js ベースの 3D モデルビューア
 * リグ付き人体モデルを読み込み、射法八節のポーズを適用する
 */

import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment } from '@react-three/drei';
import * as THREE from 'three';

interface PosedModelProps {
  boneRotations: Record<string, [number, number, number]>;
  transitionSpeed?: number;
}

/** リグ付きモデルを読み込み、ボーン回転を適用 */
function PosedModel({ boneRotations, transitionSpeed = 4 }: PosedModelProps) {
  const { scene } = useGLTF('/models/human_rigged.glb');
  const modelRef = useRef<THREE.Group>(null);
  const bonesRef = useRef<Map<string, THREE.Object3D>>(new Map());
  const targetRotationsRef = useRef<Map<string, THREE.Euler>>(new Map());

  // モデルのクローンを作成（複数インスタンスの衝突を避ける）
  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);

    // SkinnedMesh のスケルトンを修復する
    const skinnedMeshes: THREE.SkinnedMesh[] = [];
    const bones: THREE.Bone[] = [];

    clone.traverse((node) => {
      if ((node as THREE.SkinnedMesh).isSkinnedMesh) {
        skinnedMeshes.push(node as THREE.SkinnedMesh);
      }
      if ((node as THREE.Bone).isBone) {
        bones.push(node as THREE.Bone);
      }
    });

    // 元のスケルトンからボーン名とマッピングを取得
    scene.traverse((origNode) => {
      if ((origNode as THREE.SkinnedMesh).isSkinnedMesh) {
        const origMesh = origNode as THREE.SkinnedMesh;
        const origBoneNames = origMesh.skeleton.bones.map((b) => b.name);

        skinnedMeshes.forEach((cloneMesh) => {
          if (cloneMesh.name === origMesh.name) {
            const newBones = origBoneNames.map((bname) => {
              const found = bones.find((b) => b.name === bname);
              return found || new THREE.Bone();
            });
            cloneMesh.skeleton = new THREE.Skeleton(
              newBones,
              origMesh.skeleton.boneInverses.map((bi) => bi.clone())
            );
          }
        });
      }
    });

    return clone;
  }, [scene]);

  // ボーンマップを構築
  useEffect(() => {
    const map = new Map<string, THREE.Object3D>();
    clonedScene.traverse((child) => {
      if (child.name && child.name.startsWith('mixamorig:')) {
        // $AssimpFbx$ ノードを除外し、実際のボーンノードのみ取得
        if (!child.name.includes('$AssimpFbx$')) {
          map.set(child.name, child);
        }
      }
    });
    bonesRef.current = map;
  }, [clonedScene]);

  // ターゲット回転を更新
  useEffect(() => {
    const targets = new Map<string, THREE.Euler>();

    // まず全ボーンをリセット（ターゲット = identity）
    bonesRef.current.forEach((_, name) => {
      targets.set(name, new THREE.Euler(0, 0, 0));
    });

    // ポーズの回転を設定
    for (const [boneName, [x, y, z]] of Object.entries(boneRotations)) {
      targets.set(boneName, new THREE.Euler(x, y, z));
    }

    targetRotationsRef.current = targets;
  }, [boneRotations]);

  // 毎フレーム: ボーンをターゲットに向けて補間
  useFrame((_, delta) => {
    const bones = bonesRef.current;
    const targets = targetRotationsRef.current;

    targets.forEach((targetEuler, boneName) => {
      const bone = bones.get(boneName);
      if (!bone) return;

      const t = Math.min(1, delta * transitionSpeed);

      // Quaternion で SLERP 補間
      const currentQuat = new THREE.Quaternion().setFromEuler(bone.rotation);
      const targetQuat = new THREE.Quaternion().setFromEuler(targetEuler);
      currentQuat.slerp(targetQuat, t);
      bone.rotation.setFromQuaternion(currentQuat);
    });
  });

  return (
    <group ref={modelRef}>
      <primitive
        object={clonedScene}
        scale={0.018}
        position={[0, -1.5, 0]}
        rotation={[0, 0, 0]}
      />
    </group>
  );
}

/** 地面グリッド */
function Ground() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.5, 0]} receiveShadow>
      <circleGeometry args={[3, 64]} />
      <meshStandardMaterial
        color="#2a2a3a"
        transparent
        opacity={0.5}
        roughness={0.8}
      />
    </mesh>
  );
}

/** カメラの自動位置調整 */
function CameraSetup() {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 0.5, 3.5);
    camera.lookAt(0, 0, 0);
  }, [camera]);
  return null;
}

interface ModelViewerProps {
  boneRotations: Record<string, [number, number, number]>;
}

/** メインのモデルビューアコンポーネント */
const ModelViewer: React.FC<ModelViewerProps> = ({ boneRotations }) => {
  return (
    <div className="model-viewer-container">
      <Canvas
        shadows
        camera={{ fov: 45, near: 0.1, far: 100 }}
        style={{ background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)' }}
      >
        <CameraSetup />

        {/* ライティング */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[5, 8, 5]}
          intensity={1.2}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <directionalLight position={[-3, 5, -3]} intensity={0.5} color="#8888ff" />
        <pointLight position={[0, 3, 2]} intensity={0.3} color="#ffaa66" />

        {/* 環境マップ */}
        <Environment preset="studio" />

        {/* モデル */}
        <PosedModel boneRotations={boneRotations} />

        {/* 地面 */}
        <Ground />

        {/* カメラコントロール */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={1.5}
          maxDistance={8}
          target={[0, 0, 0]}
          maxPolarAngle={Math.PI * 0.85}
        />
      </Canvas>
    </div>
  );
};

// GLB のプリロード
useGLTF.preload('/models/human_rigged.glb');

export default ModelViewer;
