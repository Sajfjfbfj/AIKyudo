/**
 * 射法八節（Hassetsu）データ定義
 *
 * 各段階のポーズ（Mixamo骨格回転値）、使用筋肉、解剖学的ポイントを
 * 科学的根拠に基づき定義する。
 *
 * 参考文献：
 * - 弓道教本 第一巻〜第四巻（全日本弓道連盟）
 * - 西園(1987) 弓道立位姿勢の生体力学的研究
 * - Richardson et al. (1999) Therapeutic Exercise for Spinal Segmental Stabilization
 * - Codman (1934) The Shoulder（肩甲上腕リズム）
 * - Gray's Anatomy 42nd Edition
 * - Netter's Atlas of Human Anatomy 7th Edition
 * - Kendall et al. (2005) Muscles: Testing and Function with Posture and Pain
 * - Neumann (2010) Kinesiology of the Musculoskeletal System
 */

const D = Math.PI / 180;

/** 筋肉情報 */
export interface MuscleInfo {
  /** 筋肉名（日本語） */
  name: string;
  /** 表示色（CSS hex） */
  color: string;
  /** 役割："主" = 主動筋, "協" = 協働筋 */
  role: "主" | "協";
  /** 英語名 */
  nameEn: string;
  /** 機能説明 */
  function: string;
}

/** ポーズパラメータ（Mixamoボーン回転） */
export interface PoseAngles {
  torsoX: number;
  torsoZ: number;
  headX: number;
  headY: number;
  lShX: number;
  lShZ: number;
  lElX: number;
  lElY: number;
  lElZ: number;
  rShX: number;
  rShY: number;
  rShZ: number;
  rElX: number;
  rElY: number;
  rElZ: number;
  lScapX: number;
  lScapY: number;
  rScapX: number;
  rScapY: number;
  lHipX: number;
  lHipZ: number;
  lFootY: number;
  lKnX: number;
  rHipX: number;
  rHipZ: number;
  rFootY: number;
  rKnX: number;
}

/** 射法八節の一段階 */
export interface HassetsuStep {
  /** 段階名 */
  name: string;
  /** 段階番号ラベル */
  label: string;
  /** 説明文 */
  desc: string;
  /** 使用筋肉リスト */
  muscles: MuscleInfo[];
  /** 解剖学的ポイント */
  anatomyTags: string[];
  /** 参考文献 */
  ref: string;
  /** ポーズデータ */
  pose: PoseAngles;
}

/**
 * 射法八節 全8段階データ
 *
 * ボーン回転値:
 *   rotation(0,0,0) = T-pose（Mixamoデフォルト）
 *   LeftArm:  z+ = 上へ, x+ = 前へ
 *   RightArm: z- = 上へ, x+ = 前へ
 *   headY: + = 左向き（的方向）
 */
export const HASSETSU_STEPS: HassetsuStep[] = [
  // ① 足踏み (Ashibumi)
  {
    name: "足踏み",
    label: "①",
    desc: "基礎となる立ち方。外八文字（約60°）に踏み開くことで、骨盤を安定させ、重心を両足の間に均等に落とす。足幅は矢束（自分の矢の長さ）を基準とし、射の土台を構築する。大腿四頭筋と中殿筋が下肢を安定させ、脊柱起立筋が体幹の垂直軸を維持する。",
    muscles: [
      {
        name: "中殿筋",
        color: "#e85020",
        role: "主",
        nameEn: "Gluteus Medius",
        function: "骨盤の水平維持・股関節外転。片脚荷重時のTrendelenburg徴候を防止",
      },
      {
        name: "大腿四頭筋",
        color: "#e06010",
        role: "主",
        nameEn: "Quadriceps Femoris",
        function: "膝関節の安定伸展。微小な膝屈曲位での等尺性収縮により姿勢保持",
      },
      {
        name: "脊柱起立筋",
        color: "#2080d0",
        role: "協",
        nameEn: "Erector Spinae",
        function: "脊柱の垂直位維持。重力に抗した抗重力筋としての持続的活動",
      },
      {
        name: "前脛骨筋",
        color: "#20b080",
        role: "協",
        nameEn: "Tibialis Anterior",
        function: "足関節の安定化。足底全体での均等荷重を制御",
      },
    ],
    anatomyTags: [
      "外八文字60°",
      "骨盤：水平",
      "脊柱：垂直中立",
      "体重：左右均等",
      "足幅：矢束基準",
    ],
    ref: "弓道教本第一巻 / 西園(1987) 弓道立位姿勢の生体力学的研究",
    pose: {
      torsoX: 0, torsoZ: 0,
      headX: 0.04, headY: 0,
      lShX: 0, lShZ: -0.05, lElX: 0, lElY: 0, lElZ: 0,
      rShX: 0, rShY: 0, rShZ: 0.05, rElX: 0, rElY: 0, rElZ: 0,
      lScapX: 0, lScapY: 0, rScapX: 0, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ② 胴造り (Dozukuri)
  {
    name: "胴造り",
    label: "②",
    desc: "三重十文字（足底・腰・両肩の各線が平行かつ直角に交差する構造）を確立する。腹横筋の収縮により腹腔内圧（IAP）を高め、体幹深層筋（インナーユニット）を安定させる。多裂筋が腰椎の分節的安定性を提供し、上肢の自由な動きを支える垂直軸を形成する。",
    muscles: [
      {
        name: "腹横筋",
        color: "#e85020",
        role: "主",
        nameEn: "Transversus Abdominis",
        function: "腹腔内圧上昇による体幹安定化。呼吸相に先行して収縮し腰椎を保護",
      },
      {
        name: "多裂筋",
        color: "#4060e0",
        role: "主",
        nameEn: "Multifidus",
        function: "腰椎の分節的安定性。各椎間関節の中立位を維持する深層安定筋",
      },
      {
        name: "脊柱起立筋",
        color: "#2080d0",
        role: "協",
        nameEn: "Erector Spinae",
        function: "体幹の直立位保持。表層の抗重力筋として脊柱全体を支持",
      },
      {
        name: "腸腰筋",
        color: "#20a060",
        role: "協",
        nameEn: "Iliopsoas",
        function: "腰椎前彎の維持と骨盤前傾の制御。立位での骨盤中立位に寄与",
      },
    ],
    anatomyTags: [
      "三重十文字",
      "腰椎前彎の維持",
      "腹腔内圧(IAP)上昇",
      "骨盤中立位",
      "垂直軸の確立",
    ],
    ref: "Richardson et al. (1999) / 弓道教本第一巻 / Hodges & Richardson (1996)",
    pose: {
      torsoX: 0, torsoZ: 0,
      headX: 0.02, headY: 0,
      lShX: 0.03, lShZ: -0.07, lElX: 0, lElY: 0, lElZ: 0,
      rShX: 0.03, rShY: 0, rShZ: 0.07, rElX: 0, rElY: 0, rElZ: 0,
      lScapX: 0, lScapY: 0, rScapX: 0, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ③ 弓構え (Yugamae)
  {
    name: "弓構え",
    label: "③",
    desc: "射の準備段階。物見（頸椎C1-C2の軸回旋により頭部を的方向へ向ける）を定め、手の内（天紋筋へのグリップ密着）を形成する。前鋸筋が肩甲骨を前方に保持（プロトラクション）し、僧帽筋下部が肩甲骨を下制して肩が上がることを防ぐ。回旋筋腱板が肩関節を動的に安定させる。",
    muscles: [
      {
        name: "前鋸筋",
        color: "#e85020",
        role: "主",
        nameEn: "Serratus Anterior",
        function: "肩甲骨の前方突出（プロトラクション）と上方回旋。翼状肩甲を防止",
      },
      {
        name: "回旋筋腱板",
        color: "#9060d0",
        role: "主",
        nameEn: "Rotator Cuff",
        function: "肩関節の動的安定化。棘上筋・棘下筋・小円筋・肩甲下筋の4筋の協調収縮",
      },
      {
        name: "橈側手根屈筋",
        color: "#2080d0",
        role: "主",
        nameEn: "Flexor Carpi Radialis",
        function: "手の内形成時の手関節安定化。適度な屈曲位で弓を保持",
      },
      {
        name: "僧帽筋下部",
        color: "#20a060",
        role: "協",
        nameEn: "Lower Trapezius",
        function: "肩甲骨の下制と内転。肩の挙上（すくみ）を防止する拮抗作用",
      },
    ],
    anatomyTags: [
      "物見：頸椎C1-C2回旋",
      "手の内：天紋筋への密着",
      "肩：下制し安定させる",
      "取懸け：指の脱力",
    ],
    ref: "Netter's Atlas 7th Ed.（肩甲上腕安定機構）/ 弓道教本第一巻",
    pose: {
      torsoX: 0, torsoZ: 0,
      headX: 0, headY: 0.55,
      lShX: 0.55, lShZ: -0.40, lElX: 0.28, lElY: 0, lElZ: 0.10,
      rShX: 0.55, rShY: 0, rShZ: 0.30, rElX: 0.28, rElY: 0, rElZ: -0.10,
      lScapX: 0.04, lScapY: 0, rScapX: 0.04, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ④ 打起し (Uchiokoshi)
  {
    name: "打起し",
    label: "④",
    desc: "肩甲上腕リズム（2:1の法則）を利用して弓を正面に挙上する。三角筋前部が主動筋として肩関節を屈曲させ、前鋸筋が肩甲骨を上方回旋させることで、肩峰下インピンジメントを防ぐ。僧帽筋上部の過剰な関与を抑制し、肩がすくまないよう注意する。挙上角は約45°が基準。",
    muscles: [
      {
        name: "三角筋前部",
        color: "#e85020",
        role: "主",
        nameEn: "Anterior Deltoid",
        function: "肩関節屈曲の主動筋。0-90°の挙上で最大の筋電図活動を示す",
      },
      {
        name: "前鋸筋",
        color: "#20a060",
        role: "主",
        nameEn: "Serratus Anterior",
        function: "肩甲骨上方回旋。肩甲上腕リズムにおける肩甲骨側の運動を担当",
      },
      {
        name: "棘上筋",
        color: "#9060d0",
        role: "主",
        nameEn: "Supraspinatus",
        function: "肩関節外転の初期段階（0-30°）で始動。上腕骨頭の下方への圧迫力を提供",
      },
      {
        name: "僧帽筋中部",
        color: "#2080d0",
        role: "協",
        nameEn: "Middle Trapezius",
        function: "肩甲骨の内転（安定化）。挙上時の肩甲骨を胸郭に固定",
      },
    ],
    anatomyTags: [
      "挙上角：45°付近",
      "肩甲上腕リズム(2:1)",
      "肩を上げない",
      "僧帽筋上部の抑制",
    ],
    ref: "Codman (1934) The Shoulder / Gray's Anatomy 42nd Ed. / 弓道教本第二巻",
    pose: {
      torsoX: -0.02, torsoZ: 0,
      headX: 0, headY: 0.50,
      lShX: 0.40, lShZ: 0.65, lElX: 0.08, lElY: 0, lElZ: 0.12,
      rShX: 0.40, rShY: 0, rShZ: -0.65, rElX: 0.08, rElY: 0, rElZ: -0.12,
      lScapX: 0, lScapY: 0.12, rScapX: 0, rScapY: -0.08,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ⑤ 引分け (Hikiwake)
  {
    name: "引分け",
    label: "⑤",
    desc: "弓を左右対称に押し開く段階。腕の力ではなく、広背筋と菱形筋を用いて肩甲骨を背骨方向に寄せる（リトラクション）。押手側は上腕三頭筋と前鋸筋で矢の延長線方向に伸ばし、引手側は肘を肩の高さで後方に引く。大三から口割りへの移行を「左右均等に胸の中筋から割り込む」ように行う。",
    muscles: [
      {
        name: "広背筋",
        color: "#e85020",
        role: "主",
        nameEn: "Latissimus Dorsi",
        function: "肩関節の伸展・内転・内旋。引手側での弓を引く主要な力源",
      },
      {
        name: "菱形筋",
        color: "#9060d0",
        role: "主",
        nameEn: "Rhomboids",
        function: "肩甲骨内転（リトラクション）。肩甲骨を背骨に向けて寄せ、胸を開く",
      },
      {
        name: "上腕三頭筋",
        color: "#2080d0",
        role: "主",
        nameEn: "Triceps Brachii",
        function: "押手側の肘関節伸展。弓を的方向へ押し続ける力を発揮",
      },
      {
        name: "三角筋後部",
        color: "#20a060",
        role: "協",
        nameEn: "Posterior Deltoid",
        function: "肩関節の水平外転。引手側で肘を後方へ引く補助",
      },
    ],
    anatomyTags: [
      "大三から口割りへ",
      "肩甲骨の内転",
      "左右均等の力",
      "肘での引き",
    ],
    ref: "Netter's Atlas 7th Ed.（肩関節筋群）/ 弓道教本第二巻 / Neumann (2010)",
    pose: {
      torsoX: 0, torsoZ: 0.02,
      headX: 0, headY: 0.60,
      lShX: 0.12, lShZ: -0.08, lElX: 0.04, lElY: 0, lElZ: 0.04,
      rShX: 0.95, rShY: 0, rShZ: 0.35, rElX: 1.55, rElY: 0, rElZ: -0.30,
      lScapX: 0, lScapY: 0.08, rScapX: 0.12, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ⑥ 会 (Kai)
  {
    name: "会",
    label: "⑥",
    desc: "引分けの完成形であり、物理的な静止ではなく「無限の膨張（伸び合い・詰め合い）」の状態。広背筋と菱形筋の等尺性収縮が最大となり、胸郭を左右に押し広げることで弾性エネルギーを蓄積する。回旋筋腱板が肩関節を動的に安定させ、棘下筋が上腕の外旋位を保持する。縦横十文字（天地左右の伸び）が完成する。",
    muscles: [
      {
        name: "広背筋",
        color: "#e85020",
        role: "主",
        nameEn: "Latissimus Dorsi",
        function: "等尺性収縮による持続的な引き力の維持。エネルギー蓄積の主要筋",
      },
      {
        name: "菱形筋",
        color: "#9060d0",
        role: "主",
        nameEn: "Rhomboids",
        function: "肩甲骨内転位の維持。「詰め合い」における両肩甲骨の接近を保持",
      },
      {
        name: "回旋筋腱板",
        color: "#4060e0",
        role: "主",
        nameEn: "Rotator Cuff",
        function: "会における肩関節の動的安定化。強い外力下での関節求心力の維持",
      },
      {
        name: "棘下筋",
        color: "#e0a020",
        role: "主",
        nameEn: "Infraspinatus",
        function: "上腕骨の外旋位保持。正しい矢筋を維持するための肩関節回旋制御",
      },
    ],
    anatomyTags: [
      "縦横十文字",
      "伸び合いと詰め合い",
      "胸弦の接触",
      "気力の充実",
    ],
    ref: "Gray's Anatomy 42nd Ed. / 弓道教本第二巻 / Kendall et al. (2005)",
    pose: {
      torsoX: 0, torsoZ: 0.02,
      headX: 0, headY: 0.60,
      lShX: 0.04, lShZ: -0.06, lElX: 0.02, lElY: 0, lElZ: 0.02,
      rShX: 1.15, rShY: 0, rShZ: 0.45, rElX: 1.82, rElY: 0, rElZ: -0.40,
      lScapX: 0, lScapY: 0.04, rScapX: 0.18, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ⑦ 離れ (Hanare)
  {
    name: "離れ",
    label: "⑦",
    desc: "蓄積されたエネルギーの瞬間的な解放。作為的な指の動きではなく、「伸び合い」の延長として背筋の収縮による胸郭の広がりが臨界点に達した瞬間に、弦が自然に「はじける」現象。前鋸筋が押手を前方に送り出し、三角筋後部が引手を後方に展開する。大胸筋は胸郭の弾性復元に協働する。",
    muscles: [
      {
        name: "広背筋",
        color: "#e85020",
        role: "主",
        nameEn: "Latissimus Dorsi",
        function: "離れの瞬間の爆発的収縮。蓄積エネルギーの解放による弦の放出",
      },
      {
        name: "前鋸筋",
        color: "#9060d0",
        role: "主",
        nameEn: "Serratus Anterior",
        function: "押手側の肩甲骨前方突出。離れの瞬間に弓を的方向へ送り出す",
      },
      {
        name: "三角筋後部",
        color: "#20a060",
        role: "主",
        nameEn: "Posterior Deltoid",
        function: "引手側の肩関節水平外転。弦の解放後に右腕を後方へ展開",
      },
      {
        name: "大胸筋",
        color: "#2080d0",
        role: "協",
        nameEn: "Pectoralis Major",
        function: "胸郭の弾性復元への協働。離れの瞬間の上体安定化",
      },
    ],
    anatomyTags: [
      "自然の離れ",
      "弓返り",
      "弾性の解放",
      "両拳の鋭い離れ",
    ],
    ref: "弓道教本第二巻 / Netter's Atlas 7th Ed. / Neumann (2010)",
    pose: {
      torsoX: 0, torsoZ: 0.01,
      headX: 0, headY: 0.60,
      lShX: 0.04, lShZ: -0.22, lElX: -0.12, lElY: 0, lElZ: 0.08,
      rShX: 1.25, rShY: 0, rShZ: 0.18, rElX: 0.25, rElY: 0, rElZ: -0.08,
      lScapX: 0, lScapY: 0.04, rScapX: 0.12, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
  // ⑧ 残心 (Zanshin)
  {
    name: "残心（残身）",
    label: "⑧",
    desc: "射の完成形。離れによって生じた余韻を保ちつつ、全身の緊張を維持する段階。三角筋後部と僧帽筋中部が両腕の開いた姿勢を等尺性収縮で保持し、脊柱起立筋が体幹の垂直軸を堅持する。中殿筋が骨盤の安定性を維持し、心身一如の状態で的方向を見届ける。",
    muscles: [
      {
        name: "三角筋後部",
        color: "#20a060",
        role: "主",
        nameEn: "Posterior Deltoid",
        function: "両腕展開位の等尺性保持。離れ後の姿勢を崩さない静的筋力",
      },
      {
        name: "脊柱起立筋",
        color: "#2080d0",
        role: "主",
        nameEn: "Erector Spinae",
        function: "体幹垂直軸の堅持。離れの衝撃後も姿勢を維持する抗重力活動",
      },
      {
        name: "僧帽筋中部",
        color: "#9060d0",
        role: "主",
        nameEn: "Middle Trapezius",
        function: "肩甲骨内転位の維持。残心における「張り」の体現",
      },
      {
        name: "中殿筋",
        color: "#e85020",
        role: "協",
        nameEn: "Gluteus Medius",
        function: "骨盤水平位の維持。下肢の安定した土台を最後まで保持",
      },
    ],
    anatomyTags: [
      "姿勢の堅持",
      "気合の継続",
      "心身一如",
      "矢所の注視",
    ],
    ref: "弓道教本第一巻・第二巻 / Gray's Anatomy 42nd Ed. / Kendall et al. (2005)",
    pose: {
      torsoX: 0, torsoZ: 0,
      headX: 0, headY: 0.60,
      lShX: 0.04, lShZ: -0.22, lElX: -0.08, lElY: 0, lElZ: 0.06,
      rShX: 1.15, rShY: 0, rShZ: 0.18, rElX: 0.12, rElY: 0, rElZ: -0.06,
      lScapX: 0, lScapY: 0.04, rScapX: 0.08, rScapY: 0,
      lHipX: 0, lHipZ: 0.15, lFootY: 22 * D, lKnX: 0.02,
      rHipX: 0, rHipZ: -0.15, rFootY: -22 * D, rKnX: 0.02,
    },
  },
];

/**
 * 筋肉名 → GLBメッシュ名マッチング定義
 *
 * GLBモデルのメッシュ名（英語・日本語）と
 * UI表示名を紐付けるキーワードマッピング。
 */
export const MUSCLE_MESH_KEYWORDS: Record<string, { color: number; keys: string[] }> = {
  "中殿筋":       { color: 0xe85020, keys: ["gluteus", "medius", "中殿筋", "glut_med"] },
  "大腿四頭筋":   { color: 0xe06010, keys: ["quadriceps", "rectus", "femoris", "大腿四頭筋", "quad", "vastus"] },
  "脊柱起立筋":   { color: 0x2080d0, keys: ["erector", "spinae", "脊柱起立筋", "iliocostalis", "longissimus"] },
  "前脛骨筋":     { color: 0x20b080, keys: ["tibialis", "anterior", "前脛骨筋"] },
  "腹横筋":       { color: 0xe85020, keys: ["transversus", "abdominis", "腹横筋", "transverse"] },
  "多裂筋":       { color: 0x4060e0, keys: ["multifidus", "多裂筋"] },
  "腸腰筋":       { color: 0x20a060, keys: ["psoas", "iliacus", "腸腰筋", "iliopsoas"] },
  "前鋸筋":       { color: 0xe85020, keys: ["serratus", "前鋸筋"] },
  "回旋筋腱板":   { color: 0x9060d0, keys: ["rotator", "cuff", "supraspinatus", "infraspinatus", "teres", "subscapularis"] },
  "橈側手根屈筋": { color: 0x2080d0, keys: ["flexor", "carpi", "radialis", "橈側手根屈筋"] },
  "僧帽筋下部":   { color: 0x20a060, keys: ["trapezius", "lower", "僧帽筋下部", "trap_lower"] },
  "三角筋前部":   { color: 0xe85020, keys: ["deltoid", "anterior", "三角筋前部", "delt_ant"] },
  "棘上筋":       { color: 0x9060d0, keys: ["supraspinatus", "棘上筋"] },
  "僧帽筋中部":   { color: 0x2080d0, keys: ["trapezius", "middle", "僧帽筋中部", "trap_mid"] },
  "広背筋":       { color: 0xe85020, keys: ["latissimus", "dorsi", "広背筋", "lat"] },
  "菱形筋":       { color: 0x9060d0, keys: ["rhomboid", "菱形筋"] },
  "上腕三頭筋":   { color: 0x2080d0, keys: ["triceps", "brachii", "上腕三頭筋"] },
  "三角筋後部":   { color: 0x20a060, keys: ["deltoid", "posterior", "三角筋後部", "delt_post"] },
  "大胸筋":       { color: 0x2080d0, keys: ["pectoralis", "major", "大胸筋", "pec"] },
  "棘下筋":       { color: 0xe0a020, keys: ["infraspinatus", "棘下筋"] },
};

/**
 * Mixamoボーンマッピング定義
 *
 * ボーン名とポーズデータのプロパティの対応を定義。
 * 各ボーンにどの回転軸にどの値を適用するかを指定する。
 */
export const BONE_ROTATION_MAP: Record<string, (pose: PoseAngles) => [number, number, number]> = {
  "mixamorig:Hips":          (a) => [a.torsoX, 0, a.torsoZ],
  "mixamorig:Spine":         (a) => [a.torsoX * 0.3, 0, 0],
  "mixamorig:Spine1":        (a) => [a.torsoX * 0.3, 0, 0],
  "mixamorig:Spine2":        (a) => [a.torsoX * 0.2, 0, 0],
  "mixamorig:Neck":          (a) => [a.headX * 0.4, a.headY * 0.4, 0],
  "mixamorig:Head":          (a) => [a.headX * 0.6, a.headY * 0.6, 0],
  "mixamorig:LeftShoulder":  (a) => [a.lScapX, a.lScapY, 0],
  "mixamorig:RightShoulder": (a) => [a.rScapX, a.rScapY, 0],
  "mixamorig:LeftArm":       (a) => [a.lShX, 0, a.lShZ],
  "mixamorig:LeftForeArm":   (a) => [a.lElX, a.lElY, a.lElZ],
  "mixamorig:RightArm":      (a) => [a.rShX, a.rShY, a.rShZ],
  "mixamorig:RightForeArm":  (a) => [a.rElX, a.rElY, a.rElZ],
  "mixamorig:LeftUpLeg":     (a) => [a.lHipX, a.lFootY, a.lHipZ],
  "mixamorig:LeftLeg":       (a) => [a.lKnX, 0, 0],
  "mixamorig:RightUpLeg":    (a) => [a.rHipX, a.rFootY, a.rHipZ],
  "mixamorig:RightLeg":      (a) => [a.rKnX, 0, 0],
};
