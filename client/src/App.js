import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);

  // Handle file change from file input
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('image')) {
      setFile(selectedFile);
    } else {
      alert('Please select an image file.');
    }
  };

  // Handle file upload to backend
  const handleUpload = async () => {
    if (file) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:5000/upload', {
          method: 'POST',
          body: formData,
        });        

        if (response.ok) {
          alert('Upload successful!');
        } else {
          alert('Upload failed!');
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
      }
    } else {
      alert('Please select a file first');
    }
  };

  return (
    <div className="App" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
      <h1>Upload a Photo</h1>
      <p>Select a photo from your camera roll to upload.</p>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        capture="camera"
      />
      <button onClick={handleUpload} style={{ marginTop: '10px' }}>Upload</button>
    </div>
  );
}

export default App;
