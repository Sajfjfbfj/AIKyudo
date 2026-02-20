import React from 'react';

interface Props {
  setVideoSrc: (src: string) => void;
}

const VideoUploader: React.FC<Props> = ({ setVideoSrc }) => {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
    }
  };

  return (
    <div>
      <label htmlFor="video-upload">動画を選択:</label>
      <input
        type="file"
        id="video-upload"
        accept="video/*"
        onChange={handleFileChange}
      />
    </div>
  );
};

export default VideoUploader;