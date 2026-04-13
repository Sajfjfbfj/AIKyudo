import React, { useState } from 'react';

interface SetsuData {
  number: number;
  name: string;
  romaji: string;
  summary: string;
  pose: string[];
  points: string[];
  angles: string | null;
  emoji: string;
}

const HASSETSU: SetsuData[] = [
  {
    number: 1,
    name: '足踏み',
    romaji: 'Ashibumi',
    summary: '射の全ての土台。的に対して体を正しく据え、安定した下半身を構築する。',
    emoji: '🦶',
    pose: [
      '両足を外八文字に約60度に開く',
      '両足の親指の先と的の中心が一直線上に並ぶ',
      '足幅は自分の矢束（やづか）の長さが基準',
      '膝関節は自然に伸展させる',
      '足底で大地を踏みしめ、腰を安定させる',
    ],
    points: [
      '足踏みが広すぎると左右は安定するが前後が不安定になる',
      '足踏みが狭すぎると前後は安定するが左右が不安定になる',
      '重心は母指球〜土踏まず（踏み込みすぎない）',
      '足下を見ずに行う（視線は的方向）',
    ],
    angles: '両足の開き角度: 約60°、足幅: 矢束の長さ',
  },
  {
    number: 2,
    name: '胴造り',
    romaji: 'Douzukuri',
    summary: '足踏みの上に上半身を正しく安定させ、射の軸となる体幹を形成する。',
    emoji: '🧍',
    pose: [
      '両足に均等に体重を乗せる',
      '腰を据え、脊柱・頸部をまっすぐに伸ばす',
      '左右の肩の力を抜いて下に沈める',
      '重心を丹田（へその下）に置く',
      '弓の本弭を左膝前に置き、右手は右腰付近に',
    ],
    points: [
      '三重十文字を意識する（足底・腰・肩のラインが一直線）',
      '頭が背骨の上に乗る感覚',
      '胸を張りすぎない、骨盤は立てる（反らせない／丸めない）',
      'ひかがみ（膝の裏）をしっかり伸ばす',
      '五胴（反・屈・懸・退・中）のうち中胴が基本',
    ],
    angles: '脊柱傾き: 4°以内が理想',
  },
  {
    number: 3,
    name: '弓構え',
    romaji: 'Yugamae',
    summary: '射の準備段階。取懸け・手の内・物見の三動作で弓と身体を接続する。',
    emoji: '🏹',
    pose: [
      '取懸け: 弽（ゆがけ）の右手指を弦に正しくかける',
      '手の内: 左手で弓の握りを整える（握卵・傘を持つ手のイメージ）',
      '物見: 顔を的に向け、視線を定める',
      '右手前臂と弦が直角を保つ（懸口十文字）',
      '左右バランスを整え、弓を握るのではなく預ける感覚',
    ],
    points: [
      '正面の構えと斜面の構えの2つの形式がある',
      '握り込むと手首痛や弓返り失敗の原因',
      '物見で気を定め、的は注視せず漠然と見る',
      '呼吸を整え、気力を充実させる',
    ],
    angles: null,
  },
  {
    number: 4,
    name: '打起し',
    romaji: 'Uchiokoshi',
    summary: '弓構えの状態から両拳を頭上まで静かに持ち上げる動作。',
    emoji: '🙌',
    pose: [
      '肩に力を入れず、両腕で円を描くように柔らかく持ち上げる',
      '両拳の高さは額よりやや上、約45度の角度',
      '矢は地面と水平を保つ',
      '上体を反らさず、背中で弓を持ち上げる感覚',
      '弓のうらはずを天井に突き刺すイメージ',
    ],
    points: [
      '肩をすくめない（力任せに持ち上げると次の引分けが窮屈になる）',
      '肘・手首だけで持ち上げない',
      '45度以上高くしすぎると大三で肘先が詰まる',
      '高く遠くを意識する（低いと無理に力が入りやすい）',
    ],
    angles: '打起し角度: 約45°（体格により±5°）',
  },
  {
    number: 5,
    name: '引分け',
    romaji: 'Hikiwake',
    summary: '打起しから弓を左右に引き分け、大三を経て会に至る過程。',
    emoji: '↔️',
    pose: [
      '大三（押大目引三分一）をとる',
      '腕力ではなく背中の筋肉（肩甲骨）を使って開く',
      '左右均等にほぼ水平に引き分ける',
      '左手は押され、右手は引かれる感覚',
      '口割りより下げず、弦を胸部につける',
      '縦横十文字をつくる',
    ],
    points: [
      '大三→引分け→会は滑らかに繋がる',
      '弓は引くのではなく開くイメージ',
      '右肘を意識して弓を開いていく',
      '大三までは手の内は空回りさせ、それ以降は角見を利かせる',
      '遅速なく左右均等に引き分ける（つかみ引き・途中止まりはNG）',
    ],
    angles: '左右肩角度差: 8°以内、引き分けの滑らかさが重要',
  },
  {
    number: 6,
    name: '会',
    romaji: 'Kai',
    summary: '引分けが完了し、心身の力が最高潮に達した状態。伸び合いの極致。',
    emoji: '🎯',
    pose: [
      '矢が頬につく（頬付け）、弦が胸につく（胸弦）',
      '体と弓矢が一体となる',
      '五重十文字の完成',
      '左右への無限の伸び合い（のびあい）を続ける',
      '押し手（左肘）ピーク角度 160〜172°',
      '馬手（右肘）角度 80〜110°',
    ],
    points: [
      '静止ではなく、極めて動的な状態（永遠の引分け）',
      '呼吸を止めない',
      '弓手が潰れると矢勢が出ない',
      '会は3秒以上保つ（3秒未満は早気の可能性）',
      '詰め合いと伸び合いで矢勢を作る',
    ],
    angles: '押し手角度: 160〜172°、馬手角度: 80〜110°、会時間: 3秒以上',
  },
  {
    number: 7,
    name: '離れ',
    romaji: 'Hanare',
    summary: '会の力が満ち溢れた結果、自然に矢が放たれる瞬間。',
    emoji: '💨',
    pose: [
      '意識的に「放す」のではなく、自然に「離れる」',
      '体の中筋から左右に開くように伸長する',
      '身体の軸が崩れない',
      '肘が後方へ引ける',
      '左右が同時に外へ開く',
    ],
    points: [
      '離れは軽く鋭く（余計な力はいらない）',
      '肩をすくめて強引に放つのはNG',
      '引き腕を下げるのはNG',
      '手先で切るのはNG',
      '胸が左右に開かれることで自然に離れるのが理想',
    ],
    angles: null,
  },
  {
    number: 8,
    name: '残心',
    romaji: 'Zanshin',
    summary: '矢が離れた後の姿勢と精神を保つ、射の総決算。',
    emoji: '🧘',
    pose: [
      '離れの勢いをそのまま保ち、数秒間姿勢を崩さない',
      '縦横十文字の規矩を堅持する',
      '矢所（矢の行方）を注視する',
      '弓返りが自然に起きる',
      '左右の伸びが残っている',
    ],
    points: [
      '残心は射の答え合わせ（技術検証のフェーズ）',
      '美しい残心は正しい射の証',
      '離れて気合をぬかず十分伸び合い、弓倒しをする',
      '弓返りの自然さ、身体の軸の立ち方、離れの方向性を確認',
      '精神姿勢だけでなく技術検証でもある',
    ],
    angles: null,
  },
];

const ShahouHassetsu: React.FC = () => {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  return (
    <div className="hassetsu-section">
      <h3 className="hassetsu-title">射法八節 リファレンス</h3>
      <p className="hassetsu-subtitle">
        弓道の基本動作を8段階に分けた射法八節。各節のポージングと要点を確認できます。
      </p>

      <div className="hassetsu-flow">
        {HASSETSU.map((s, i) => (
          <React.Fragment key={s.number}>
            <button
              className={`hassetsu-flow-node${openIdx === i ? ' active' : ''}`}
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
              title={s.name}
            >
              <span className="hassetsu-flow-emoji">{s.emoji}</span>
              <span className="hassetsu-flow-name">{s.name}</span>
            </button>
            {i < HASSETSU.length - 1 && (
              <span className="hassetsu-flow-arrow">→</span>
            )}
          </React.Fragment>
        ))}
      </div>

      <div className="hassetsu-cards">
        {HASSETSU.map((s, i) => (
          <div
            className={`hassetsu-card${openIdx === i ? ' open' : ''}`}
            key={s.number}
          >
            <button
              className="hassetsu-card-header"
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
            >
              <div className="hassetsu-card-num">{s.number}</div>
              <div className="hassetsu-card-info">
                <span className="hassetsu-card-name">
                  {s.emoji} {s.name}
                </span>
                <span className="hassetsu-card-romaji">{s.romaji}</span>
              </div>
              <span className="hassetsu-card-toggle">
                {openIdx === i ? '▲' : '▼'}
              </span>
            </button>

            {openIdx === i && (
              <div className="hassetsu-card-body">
                <p className="hassetsu-card-summary">{s.summary}</p>

                <div className="hassetsu-card-group">
                  <h4>ポージング（体の形）</h4>
                  <ul>
                    {s.pose.map((p, j) => (
                      <li key={j}>{p}</li>
                    ))}
                  </ul>
                </div>

                <div className="hassetsu-card-group">
                  <h4>注意点・コツ</h4>
                  <ul>
                    {s.points.map((p, j) => (
                      <li key={j}>{p}</li>
                    ))}
                  </ul>
                </div>

                {s.angles && (
                  <div className="hassetsu-card-angles">
                    <strong>理想角度・基準値：</strong>
                    {s.angles}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ShahouHassetsu;
