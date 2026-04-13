import React, { useState } from 'react';
import VideoUploader from './components/VideoUploader';
import VideoAnalyzer from './components/VideoAnalyzer';
import HassetsuPage from './components/HassetsuPage';
import './App.css';

type Page = 'analyzer' | 'hassetsu';

function App() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [page, setPage] = useState<Page>('analyzer');

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <span className="header-icon">🏹</span>
          <div>
            <h1 className="app-title">弓道 AI フォーム解析</h1>
            <p className="app-subtitle">Kyudo Form Analyzer — MediaPipe Pose</p>
          </div>
          <nav className="header-nav">
            <button
              className={`nav-tab ${page === 'analyzer' ? 'active' : ''}`}
              onClick={() => setPage('analyzer')}
            >
              フォーム解析
            </button>
            <button
              className={`nav-tab ${page === 'hassetsu' ? 'active' : ''}`}
              onClick={() => setPage('hassetsu')}
            >
              射法八節 3D
            </button>
          </nav>
        </div>
      </header>

      {page === 'analyzer' && (
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
        </main>
      )}

      {page === 'hassetsu' && <HassetsuPage />}
    </div>
  );
}

export default App;
