/**
 * HassetsuPage.tsx
 * 射法八節ポーズビューアのメインページ
 * 3Dモデルと各段階の解説を表示し、ナビゲーションで切り替える
 */

import React, { useState } from 'react';
import ModelViewer from './ModelViewer';
import { hassetsuStages } from '../data/hassetsuPoses';

const HassetsuPage: React.FC = () => {
  const [currentStage, setCurrentStage] = useState(0);
  const stage = hassetsuStages[currentStage];

  const goToPrev = () =>
    setCurrentStage((prev) => (prev > 0 ? prev - 1 : hassetsuStages.length - 1));
  const goToNext = () =>
    setCurrentStage((prev) => (prev < hassetsuStages.length - 1 ? prev + 1 : 0));

  return (
    <div className="hassetsu-page">
      {/* 3Dビューア */}
      <div className="hassetsu-viewer">
        <ModelViewer boneRotations={stage.boneRotations} />

        {/* ステージインジケーター（3Dビューア上にオーバーレイ） */}
        <div className="stage-indicator">
          <span className="stage-number">{stage.id}</span>
          <span className="stage-divider">/</span>
          <span className="stage-total">8</span>
        </div>

        {/* 操作ヒント */}
        <div className="viewer-hint">
          ドラッグで回転 ・ スクロールでズーム
        </div>
      </div>

      {/* 情報パネル */}
      <div className="hassetsu-info">
        {/* ナビゲーション */}
        <div className="stage-nav">
          <button className="nav-btn" onClick={goToPrev} aria-label="前の段階">
            ◀
          </button>

          <div className="stage-title-block">
            <h2 className="stage-name-ja">{stage.nameJa}</h2>
            <p className="stage-name-en">
              {stage.nameEn}（{stage.reading}）
            </p>
          </div>

          <button className="nav-btn" onClick={goToNext} aria-label="次の段階">
            ▶
          </button>
        </div>

        {/* 説明文 */}
        <div className="stage-description">
          <p>{stage.description}</p>
        </div>

        {/* ステージ選択ドット */}
        <div className="stage-dots">
          {hassetsuStages.map((s, i) => (
            <button
              key={s.id}
              className={`stage-dot ${i === currentStage ? 'active' : ''}`}
              onClick={() => setCurrentStage(i)}
              aria-label={`${s.nameJa}を表示`}
              title={`${s.id}. ${s.nameJa}`}
            >
              <span className="dot-label">{s.nameJa}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HassetsuPage;
