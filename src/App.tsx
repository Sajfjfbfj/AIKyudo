import React, { useState } from 'react';
import VideoUploader from './components/VideoUploader';
import VideoAnalyzer from './components/VideoAnalyzer';
import ShahouHassetsu from './components/ShahouHassetsu';
import './App.css';

function App() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <span className="header-icon">🏹</span>
          <div>
            <h1 className="app-title">弓道 AI フォーム解析</h1>
            <p className="app-subtitle">Kyudo Form Analyzer — MediaPipe Pose</p>
          </div>
        </div>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <VideoUploader setVideoSrc={setVideoSrc} />
          {!videoSrc && (
            <p className="upload-hint">
              動画をアップロードすると自動で骨格解析・フォーム評価が始まります
            </p>
          )}
        </section>

        {videoSrc && (
          <section className="analysis-section">
            <VideoAnalyzer videoSrc={videoSrc} />
          </section>
        )}

        <ShahouHassetsu />
      </main>
    </div>
  );
}

export default App;
