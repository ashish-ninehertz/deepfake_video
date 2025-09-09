import React, { useState, useCallback } from 'react'
import { Upload, AlertCircle, CheckCircle, XCircle, Loader, Video, Shield, Eye, Image as ImageIcon, X, ZoomIn } from 'lucide-react'
import axios from 'axios'

const App = () => {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [selectedFrame, setSelectedFrame] = useState(null)

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      // Check file size (50 MB limit)
      const maxSize = 50 * 1024 * 1024 // 50 MB in bytes
      if (selectedFile.size > maxSize) {
        setError(`File size exceeds 50 MB limit. Your file is ${formatFileSize(selectedFile.size)}.`)
        return
      }
      
      setFile(selectedFile)
      setResult(null)
      setError(null)
    } else {
      setError('Please select a valid video file')
    }
  }

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const droppedFile = e.dataTransfer.files[0]
    handleFileSelect(droppedFile)
  }, [])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    handleFileSelect(selectedFile)
  }

  const analyzeVideo = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('video', file)

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const resetUpload = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setSelectedFrame(null)
  }

  const openFrameModal = (frameBase64, index) => {
    setSelectedFrame({ image: frameBase64, index })
  }

  const closeFrameModal = () => {
    setSelectedFrame(null)
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4 rounded-full">
              <Shield className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Deepfake Detection System
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a video to analyze whether it contains deepfake content using advanced AI detection
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {/* Upload Section */}
          <div className="card mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
              <Upload className="w-6 h-6 mr-2" />
              Upload Video
            </h2>

            {!file ? (
              <div
                className={`upload-area border-2 border-dashed border-gray-300 rounded-xl p-12 text-center transition-all duration-300 ${
                  dragOver ? 'dragover border-blue-500' : ''
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="text-white mb-4">
                  <Video className="w-16 h-16 mx-auto mb-4 opacity-80" />
                  <h3 className="text-xl font-semibold mb-2">Drop your video here</h3>
                  <p className="text-blue-100">or click to browse</p>
                  <p className="text-blue-200 text-sm mt-2">
                    Supports MP4, AVI, MOV, MKV files • Max size: 50 MB
                  </p>
                </div>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <Video className="w-8 h-8 text-blue-600 mr-3" />
                    <div>
                      <h3 className="font-semibold text-gray-800">{file.name}</h3>
                      <p className="text-sm text-gray-600">{formatFileSize(file.size)}</p>
                    </div>
                  </div>
                  <button
                    onClick={resetUpload}
                    className="text-red-600 hover:text-red-800 font-medium"
                  >
                    Remove
                  </button>
                </div>

                <button
                  onClick={analyzeVideo}
                  disabled={loading}
                  className={`w-full btn-primary ${
                    loading ? 'opacity-50 cursor-not-allowed' : ''
                  } flex items-center justify-center`}
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing Video...
                    </>
                  ) : (
                    <>
                      <Eye className="w-5 h-5 mr-2" />
                      Analyze Video
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="card mb-8 bg-red-50 border-red-200">
              <div className="flex items-center text-red-800">
                <AlertCircle className="w-6 h-6 mr-3 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold">Analysis Failed</h3>
                  <p className="text-sm mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results Section */}
          {result && (
            <div className={`card ${result.prediction === 'REAL' ? 'result-real' : 'result-fake'}`}>
              <div className="text-center">
                <div className="mb-4">
                  {result.prediction === 'REAL' ? (
                    <CheckCircle className="w-16 h-16 mx-auto text-real" />
                  ) : (
                    <XCircle className="w-16 h-16 mx-auto text-fake" />
                  )}
                </div>

                <h2 className="text-3xl font-bold mb-2">
                  {result.prediction === 'REAL' ? 'AUTHENTIC VIDEO' : 'DEEPFAKE DETECTED'}
                </h2>

                <div className="text-lg mb-6">
                  <p className="mb-2">
                    Confidence: <span className="font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                  </p>
                  <p className="text-sm opacity-75">
                    Frames analyzed: {result.frames_extracted}
                  </p>
                </div>

                <div className="bg-white/20 rounded-lg p-4">
                  <p className="text-sm font-medium">{result.message}</p>
                </div>
              </div>
            </div>
          )}

          {/* Extracted Frames Section */}
          {result && result.frames && result.frames.length > 0 && (
            <div className="card">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                <ImageIcon className="w-6 h-6 mr-2" />
                Extracted Frames Used for Analysis
              </h2>
              
              <p className="text-gray-600 mb-6">
                These are the {result.frames.length} frames extracted from your video using motion detection and face recognition. 
                Our AI model analyzed these frames to make the prediction.
              </p>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {result.frames.map((frameBase64, index) => (
                  <div key={index} className="relative group cursor-pointer" onClick={() => openFrameModal(frameBase64, index)}>
                    <img
                      src={frameBase64}
                      alt={`Extracted frame ${index + 1}`}
                      className="w-full h-24 object-cover rounded-lg border-2 border-gray-200 group-hover:border-blue-400 transition-colors duration-200"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 rounded-lg transition-all duration-200 flex items-center justify-center">
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 text-center">
                        <ZoomIn className="w-6 h-6 text-white mx-auto mb-1" />
                        <span className="text-white text-xs font-medium">
                          Frame {index + 1}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 text-sm text-gray-500 text-center">
                Click on any frame to view it larger
              </div>
            </div>
          )}

          {/* Loading Animation */}
          {loading && (
            <div className="card text-center">
              <div className="animate-pulse-slow mb-4">
                <Shield className="w-16 h-16 mx-auto text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                Analyzing Video...
              </h3>
              <p className="text-gray-600 mb-4">
                Our AI is examining the video for deepfake indicators
              </p>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full animate-pulse"></div>
              </div>
            </div>
          )}

          {/* Info Section */}
          <div className="mt-12 grid md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-white rounded-lg shadow-sm">
              <Shield className="w-8 h-8 mx-auto text-blue-600 mb-3" />
              <h3 className="font-semibold text-gray-800 mb-2">AI-Powered</h3>
              <p className="text-sm text-gray-600">
                Advanced deep learning models trained on extensive datasets
              </p>
            </div>
            <div className="text-center p-6 bg-white rounded-lg shadow-sm">
              <Eye className="w-8 h-8 mx-auto text-purple-600 mb-3" />
              <h3 className="font-semibold text-gray-800 mb-2">Frame Analysis</h3>
              <p className="text-sm text-gray-600">
                Analyzes multiple frames with motion detection for accuracy
              </p>
            </div>
            <div className="text-center p-6 bg-white rounded-lg shadow-sm">
              <CheckCircle className="w-8 h-8 mx-auto text-green-600 mb-3" />
              <h3 className="font-semibold text-gray-800 mb-2">High Accuracy</h3>
              <p className="text-sm text-gray-600">
                Trained on real and synthetic video data for reliable detection
              </p>
            </div>
          </div>
        </div>

        {/* Frame Modal */}
        {selectedFrame && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4" onClick={closeFrameModal}>
            <div className="relative max-w-4xl max-h-full" onClick={(e) => e.stopPropagation()}>
              <button
                onClick={closeFrameModal}
                className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors duration-200"
              >
                <X className="w-8 h-8" />
              </button>
              <img
                src={selectedFrame.image}
                alt={`Frame ${selectedFrame.index + 1} - Full Size`}
                className="max-w-full max-h-full rounded-lg shadow-2xl"
              />
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4 rounded-b-lg">
                <p className="text-white text-center font-medium">
                  Frame {selectedFrame.index + 1} - Extracted for Analysis
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
