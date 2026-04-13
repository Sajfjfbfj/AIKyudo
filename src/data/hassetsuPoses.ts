/**
 * 射法八節（しゃほうはっせつ）ポーズ定義
 *
 * Mixamo リグのボーン回転値を定義する。
 * 各ボーンの回転は Euler (x, y, z) ラジアンで、rest pose (T-ポーズ) からの差分。
 *
 * 座標系:
 *   Y-up, 右手系 (glTF 標準)
 *   キャラクターは +Z 方向を向いている
 *   的は左側 (-X 方向)
 */

export interface HassetsuStage {
  id: number;
  nameJa: string;
  nameEn: string;
  reading: string;
  description: string;
  /** ボーン名 → [x, y, z] Euler ラジアン */
  boneRotations: Record<string, [number, number, number]>;
}

const PI = Math.PI;
const DEG = PI / 180;

/**
 * 射法八節の8段階定義
 */
export const hassetsuStages: HassetsuStage[] = [
  // ──────────────────────────────────────────────
  // 1. 足踏み（あしぶみ）— Footing
  // ──────────────────────────────────────────────
  {
    id: 1,
    nameJa: '足踏み',
    nameEn: 'Ashibumi',
    reading: 'あしぶみ',
    description:
      '射の土台を作る動作。両足を的の中心に向かって外八文字（約60°）に踏み開き、矢束の幅で安定した下半身を構築する。重心は両足の中心に置き、体軸をまっすぐに保つ。',
    boneRotations: {
      // 両腕は体側に自然に下ろす（弓を左手に持つ）
      'mixamorig:LeftArm': [0.15, 0, -55 * DEG],
      'mixamorig:LeftForeArm': [0, -0.25, -10 * DEG],
      'mixamorig:LeftHand': [0.1, 0, 0],
      'mixamorig:RightArm': [-0.15, 0, 55 * DEG],
      'mixamorig:RightForeArm': [0, 0.25, 10 * DEG],
      'mixamorig:RightHand': [-0.1, 0, 0],
      // 足を外八文字に開く
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋をまっすぐに
      'mixamorig:Spine': [-0.02, 0, 0],
      'mixamorig:Spine1': [-0.02, 0, 0],
      'mixamorig:Spine2': [-0.02, 0, 0],
    },
  },

  // ──────────────────────────────────────────────
  // 2. 胴造り（どうづくり）— Body Formation
  // ──────────────────────────────────────────────
  {
    id: 2,
    nameJa: '胴造り',
    nameEn: 'Dozukuri',
    reading: 'どうづくり',
    description:
      '足踏みの上に上半身を正しく安定させる。脊柱をまっすぐに伸ばし、左右の肩の力を抜き、重心を丹田（へその下）に置く。三重十文字（肩・腰・足底のライン）が整った状態を作る。',
    boneRotations: {
      // 両腕は体側、左手は弓の本弭を左膝頭に置く
      'mixamorig:LeftArm': [0.2, 0, -60 * DEG],
      'mixamorig:LeftForeArm': [-0.15, -0.35, -15 * DEG],
      'mixamorig:LeftHand': [0.15, -0.1, 0],
      'mixamorig:RightArm': [-0.1, 0, 58 * DEG],
      'mixamorig:RightForeArm': [0, 0.2, 8 * DEG],
      'mixamorig:RightHand': [-0.05, 0, 0],
      // 足は足踏みと同じ
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋をまっすぐに、肩を下げる
      'mixamorig:Spine': [-0.03, 0, 0],
      'mixamorig:Spine1': [-0.02, 0, 0],
      'mixamorig:Spine2': [-0.02, 0, 0],
      'mixamorig:LeftShoulder': [0, 0, -0.05],
      'mixamorig:RightShoulder': [0, 0, 0.05],
    },
  },

  // ──────────────────────────────────────────────
  // 3. 弓構え（ゆがまえ）— Readying the Bow
  // ──────────────────────────────────────────────
  {
    id: 3,
    nameJa: '弓構え',
    nameEn: 'Yugamae',
    reading: 'ゆがまえ',
    description:
      '射の準備段階。「取懸け」で右手の指を弦にかけ、「手の内」で左手の弓の握りを整え、「物見」で顔を的に向ける。両腕で円相（えんそう）を作り、心身を射に向けて整える。',
    boneRotations: {
      // 物見: 顔を左（的方向）に向ける
      'mixamorig:Head': [0, 35 * DEG, 0],
      'mixamorig:Neck': [0, 10 * DEG, 0],
      // 両腕を前に出し円相を作る
      'mixamorig:LeftArm': [25 * DEG, 0, -40 * DEG],
      'mixamorig:LeftForeArm': [0, -55 * DEG, -20 * DEG],
      'mixamorig:LeftHand': [0.2, -0.2, 0.1],
      'mixamorig:RightArm': [-25 * DEG, 0, 40 * DEG],
      'mixamorig:RightForeArm': [0, 55 * DEG, 20 * DEG],
      'mixamorig:RightHand': [-0.2, 0.2, -0.1],
      // 足は同じ
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋
      'mixamorig:Spine': [-0.03, 0, 0],
      'mixamorig:Spine1': [-0.02, 0, 0],
      'mixamorig:Spine2': [-0.02, 0, 0],
    },
  },

  // ──────────────────────────────────────────────
  // 4. 打起し（うちおこし）— Raising the Bow
  // ──────────────────────────────────────────────
  {
    id: 4,
    nameJa: '打起し',
    nameEn: 'Uchiokoshi',
    reading: 'うちおこし',
    description:
      '弓構えの状態から、両拳を額より上（約45度）まで静かに持ち上げる。肩の力を入れず、両腕で円を描くように柔らかく弓矢を持ち上げる。矢は地面と水平を保つ。',
    boneRotations: {
      // 物見を維持
      'mixamorig:Head': [0, 35 * DEG, 0],
      'mixamorig:Neck': [0, 10 * DEG, 0],
      // 両腕を頭上に上げる (約45度以上)
      'mixamorig:LeftArm': [20 * DEG, 15 * DEG, 55 * DEG],
      'mixamorig:LeftForeArm': [0, -25 * DEG, 10 * DEG],
      'mixamorig:LeftHand': [0.1, -0.15, 0.1],
      'mixamorig:RightArm': [-20 * DEG, -15 * DEG, -55 * DEG],
      'mixamorig:RightForeArm': [0, 25 * DEG, -10 * DEG],
      'mixamorig:RightHand': [-0.1, 0.15, -0.1],
      // 足
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋 — 肩を上げない
      'mixamorig:Spine': [-0.03, 0, 0],
      'mixamorig:Spine1': [-0.02, 0, 0],
      'mixamorig:Spine2': [-0.02, 0, 0],
      'mixamorig:LeftShoulder': [0, 0, -0.05],
      'mixamorig:RightShoulder': [0, 0, 0.05],
    },
  },

  // ──────────────────────────────────────────────
  // 5. 引分け（ひきわけ）— Drawing
  // ──────────────────────────────────────────────
  {
    id: 5,
    nameJa: '引分け',
    nameEn: 'Hikiwake',
    reading: 'ひきわけ',
    description:
      '打起しから弓を左右に引き分ける動作。まず大三（だいさん）を取り、左腕で弓を押し開き、右手は弦に引かれるように体の中心へ。背中の筋肉を使い、肩甲骨を動かして力強く引き分ける。',
    boneRotations: {
      // 物見を維持
      'mixamorig:Head': [0, 35 * DEG, 0],
      'mixamorig:Neck': [0, 10 * DEG, 0],
      // 左腕: 的方向に押し出す（大三の位置）
      'mixamorig:LeftArm': [15 * DEG, 35 * DEG, 30 * DEG],
      'mixamorig:LeftForeArm': [0, -15 * DEG, 5 * DEG],
      'mixamorig:LeftHand': [0.05, -0.15, 0.15],
      // 右腕: 引き戻す、肘を曲げる
      'mixamorig:RightArm': [-30 * DEG, -30 * DEG, -10 * DEG],
      'mixamorig:RightForeArm': [0, 80 * DEG, -5 * DEG],
      'mixamorig:RightHand': [-0.15, 0.1, -0.2],
      // 足
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋 — 胸を開く
      'mixamorig:Spine': [-0.03, 0, 0],
      'mixamorig:Spine1': [-0.03, 0, 0],
      'mixamorig:Spine2': [-0.03, 0, 0],
    },
  },

  // ──────────────────────────────────────────────
  // 6. 会（かい）— Full Draw
  // ──────────────────────────────────────────────
  {
    id: 6,
    nameJa: '会',
    nameEn: 'Kai',
    reading: 'かい',
    description:
      '引分けが完了した状態。矢が頬につき（頬付け）、弦が胸につく（胸弦）。左腕は的方向に完全に伸び、右手は口の高さ（口割り）に収まる。静止ではなく、左右への無限の伸び合いが続く動的な状態。',
    boneRotations: {
      // 物見を維持
      'mixamorig:Head': [0, 35 * DEG, 0],
      'mixamorig:Neck': [0, 10 * DEG, 0],
      // 左腕: 的方向に完全に伸ばす
      'mixamorig:LeftArm': [5 * DEG, 40 * DEG, 10 * DEG],
      'mixamorig:LeftForeArm': [0, -8 * DEG, 3 * DEG],
      'mixamorig:LeftHand': [0, -0.15, 0.2],
      // 右腕: 完全に引ききる — 右手は口の高さ、右肘は後方
      'mixamorig:RightArm': [-50 * DEG, -55 * DEG, 15 * DEG],
      'mixamorig:RightForeArm': [10 * DEG, 110 * DEG, 0],
      'mixamorig:RightHand': [-0.2, 0.15, -0.3],
      // 足
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋 — 胸を大きく開く
      'mixamorig:Spine': [-0.04, 0, 0],
      'mixamorig:Spine1': [-0.03, 0, 0],
      'mixamorig:Spine2': [-0.03, 0, 0],
    },
  },

  // ──────────────────────────────────────────────
  // 7. 離れ（はなれ）— Release
  // ──────────────────────────────────────────────
  {
    id: 7,
    nameJa: '離れ',
    nameEn: 'Hanare',
    reading: 'はなれ',
    description:
      '会の伸び合いが満ちた結果として、自然に矢が放たれる瞬間。意識的に「放す」のではなく、胸が左右に開かれることで「離れる」。左右均等に鋭い離れが理想。',
    boneRotations: {
      // 物見を維持
      'mixamorig:Head': [0, 30 * DEG, 0],
      'mixamorig:Neck': [0, 10 * DEG, 0],
      // 左腕: 的方向に伸びたまま
      'mixamorig:LeftArm': [0, 40 * DEG, 5 * DEG],
      'mixamorig:LeftForeArm': [0, -5 * DEG, 3 * DEG],
      'mixamorig:LeftHand': [0, -0.1, 0.25],
      // 右腕: 離れの動きで右後方に開く
      'mixamorig:RightArm': [-25 * DEG, -70 * DEG, 30 * DEG],
      'mixamorig:RightForeArm': [10 * DEG, 60 * DEG, 0],
      'mixamorig:RightHand': [-0.1, 0.1, -0.2],
      // 足
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋 — 胸を開いた状態を維持
      'mixamorig:Spine': [-0.04, 0, 0],
      'mixamorig:Spine1': [-0.03, 0, 0],
      'mixamorig:Spine2': [-0.03, 0, 0],
    },
  },

  // ──────────────────────────────────────────────
  // 8. 残心（ざんしん）— Follow-through
  // ──────────────────────────────────────────────
  {
    id: 8,
    nameJa: '残心',
    nameEn: 'Zanshin',
    reading: 'ざんしん',
    description:
      '矢が離れた後の姿勢と精神をそのまま保つ、射の総決算。離れの勢いをそのままに、両腕を大きく左右に開いた「大の字」に近い形を数秒間保つ。美しい残心は正しい射の証。',
    boneRotations: {
      // 物見を維持（的を見届ける）
      'mixamorig:Head': [0, 30 * DEG, 0],
      'mixamorig:Neck': [0, 8 * DEG, 0],
      // 左腕: 大きく左に開く
      'mixamorig:LeftArm': [-5 * DEG, 35 * DEG, 15 * DEG],
      'mixamorig:LeftForeArm': [0, -5 * DEG, 2 * DEG],
      'mixamorig:LeftHand': [0, -0.1, 0.2],
      // 右腕: 大きく右に開く（大の字）
      'mixamorig:RightArm': [5 * DEG, -35 * DEG, -15 * DEG],
      'mixamorig:RightForeArm': [0, 15 * DEG, -2 * DEG],
      'mixamorig:RightHand': [0, 0.1, -0.1],
      // 足
      'mixamorig:LeftUpLeg': [0.05, 0.2, 0],
      'mixamorig:RightUpLeg': [0.05, -0.2, 0],
      'mixamorig:LeftFoot': [0, 0.25, 0],
      'mixamorig:RightFoot': [0, -0.25, 0],
      // 背筋 — 体軸を保つ
      'mixamorig:Spine': [-0.03, 0, 0],
      'mixamorig:Spine1': [-0.02, 0, 0],
      'mixamorig:Spine2': [-0.02, 0, 0],
    },
  },
];
