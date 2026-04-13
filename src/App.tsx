import React, { useState } from 'react';
import VideoUploader from './components/VideoUploader';
import VideoAnalyzer from './components/VideoAnalyzer';
import HassetsuViewer from './components/HassetsuViewer';
import './App.css';
import './components/HassetsuViewer.css';

type Page = 'analyzer' | 'hassetsu';

function App() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [page, setPage] = useState<Page>('hassetsu');

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
        <nav className="header-nav">
          <button
            className={`nav-btn${page === 'hassetsu' ? ' active' : ''}`}
            onClick={() => setPage('hassetsu')}
          >
            射法八節 3D
          </button>
          <button
            className={`nav-btn${page === 'analyzer' ? ' active' : ''}`}
            onClick={() => setPage('analyzer')}
          >
            動画解析
          </button>
        </nav>
      </header>

      <main className="app-main">
        {page === 'hassetsu' && (
          <section className="analysis-section">
            <HassetsuViewer />
          </section>
        )}

        {page === 'analyzer' && (
          <>
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
          </>
        )}
      </main>
    </div>
  );
}

export default App;
