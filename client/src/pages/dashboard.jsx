import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Upload,
  Brain,
  BarChart3,
  History,
  RefreshCw,
  Activity,
  FileAudio,
  Zap,
  TrendingUp,
  Database,
  Play,
  CheckCircle,
  XCircle,
  AlertCircle,
  Settings,
  Monitor,
  Mic,
  Square,
  Trash2,
  Plus,
  Minus
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // File uploads
  const [predictFile, setPredictFile] = useState(null);
  const [retrainFiles, setRetrainFiles] = useState([]);
  const [logFile, setLogFile] = useState(null);
  const [retrainLabel, setRetrainLabel] = useState('');
  const [logLabel, setLogLabel] = useState('');

  // Audio recording
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);

  useEffect(() => {
    fetchModelInfo();
    fetchTrainingHistory();
    fetchPredictionHistory();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model_info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchTrainingHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/training_history?limit=20`);
      setTrainingHistory(response.data.training_history);
    } catch (error) {
      console.error('Error fetching training history:', error);
    }
  };

  const fetchPredictionHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/prediction_history?limit=20`);
      setPredictionHistory(response.data.prediction_history);
    } catch (error) {
      console.error('Error fetching prediction history:', error);
    }
  };
  const startRecording = async () => {
    try {
      // Stop any existing recording
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }

      // Get audio stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

      // Create MediaRecorder
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });
      audioChunksRef.current = [];

      // Handle data available
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Handle recording stop
      mediaRecorderRef.current.onstop = async () => {
        try {
          // Combine all chunks
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

          // Convert webm to wav
          const wavBlob = await convertWebmToWav(audioBlob);

          // Create file object
          const audioFile = new File([wavBlob], `recording_${Date.now()}.wav`, {
            type: 'audio/wav'
          });

          // Update state
          setRecordedAudio(audioFile);
          setPredictFile(audioFile);

          // Create preview URL
          setAudioPreviewUrl(URL.createObjectURL(wavBlob));
        } catch (error) {
          console.error('Error processing recording:', error);
        } finally {
          // Clean up
          stream.getTracks().forEach(track => track.stop());
        }
      };

      // Start recording
      mediaRecorderRef.current.start(100); // Collect data every 100ms
      setIsRecording(true);
      setRecordingDuration(0);

      // Start duration timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error starting recording:', error);
      setIsRecording(false);
      clearInterval(recordingIntervalRef.current);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingIntervalRef.current);
    }
  };

  // New function to handle WebM to WAV conversion
  async function convertWebmToWav(webmBlob) {
    // Step 1: Decode the WebM audio
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000
    });
    const decodedData = await audioContext.decodeAudioData(arrayBuffer);

    // Step 2: Get PCM data
    const pcmData = decodedData.getChannelData(0); // Get mono channel

    // Step 3: Convert to WAV
    return encodeWAV(pcmData, audioContext.sampleRate, 1);
  }

  // Updated WAV encoder
  function encodeWAV(samples, sampleRate, numChannels) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // Format chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);

    // Data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Convert samples to 16-bit PCM
    const volume = 1;
    let index = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      index += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }


  const clearRecording = () => {
    setRecordedAudio(null);
    setPredictFile(null);
    setRecordingDuration(0);
  };

  const addRetrainFile = () => {
    const newFile = {
      id: Date.now(),
      file: null,
      label: ''
    };
    setRetrainFiles([...retrainFiles, newFile]);
  };

  const removeRetrainFile = (id) => {
    setRetrainFiles(retrainFiles.filter(file => file.id !== id));
  };

  const updateRetrainFile = (id, field, value) => {
    setRetrainFiles(retrainFiles.map(file =>
      file.id === id ? { ...file, [field]: value } : file
    ));
  };

  const canRetrain = () => {
    const validFiles = retrainFiles.filter(file => file.file && file.label);
    return validFiles.length >= 10 && validFiles.length % 10 === 0;
  };

  const handlePredict = async () => {
    if (!predictFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', predictFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });
      setPrediction(response.data);
      fetchPredictionHistory();
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const handleRetrain = async () => {
    const validFiles = retrainFiles.filter(file => file.file && file.label);
    if (!canRetrain()) return;

    setLoading(true);

    try {
      // Send files in batches or individually - depending on your API design
      for (const fileData of validFiles) {
        const formData = new FormData();
        formData.append('file', fileData.file);
        formData.append('label', fileData.label);

        await axios.post(`${API_BASE_URL}/retrain`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      }

      alert(`Model retrained successfully with ${validFiles.length} files!`);
      setRetrainFiles([]);
      fetchTrainingHistory();
      fetchModelInfo();
    } catch (error) {
      console.error('Retrain error:', error);
      alert('Error during retraining. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogPrediction = async () => {
    if (!logFile || !logLabel) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', logFile);
    formData.append('true_label', logLabel);

    try {
      await axios.post(`${API_BASE_URL}/log_prediction`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert('Prediction logged successfully!');
      fetchPredictionHistory();
    } catch (error) {
      console.error('Log prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetMetrics = async () => {
    try {
      await axios.post(`${API_BASE_URL}/reset_metrics`);
      setMetrics(null);
      alert('Metrics reset successfully!');
    } catch (error) {
      console.error('Reset metrics error:', error);
    }
  };

  const TabButton = ({ id, label, icon: Icon, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all duration-200 ${isActive
        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }`}
    >
      <Icon size={20} />
      <span className="font-medium">{label}</span>
    </button>
  );

  const StatCard = ({ title, value, icon: Icon, color, subtitle }) => (
    <div className="bg-white rounded-xl p-6 border border-gray-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-lg bg-gradient-to-br ${color.includes('blue') ? 'from-blue-50 to-blue-100' : color.includes('green') ? 'from-green-50 to-green-100' : 'from-purple-50 to-purple-100'}`}>
          <Icon className={color} size={24} />
        </div>
      </div>
    </div>
  );

  const FileUpload = ({ file, setFile, accept, label, disabled }) => (
    <div className="border border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
      <FileAudio className="mx-auto text-gray-400 mb-2" size={32} />
      <label className="cursor-pointer">
        <span className="text-sm font-medium text-blue-600 hover:text-blue-500">
          {file ? file.name : `Choose ${label}`}
        </span>
        <input
          type="file"
          accept={accept}
          onChange={(e) => setFile(e.target.files[0])}
          disabled={disabled}
          className="hidden"
        />
      </label>
      <p className="text-xs text-gray-500 mt-1">WAV files only</p>
    </div>
  );

  const AudioRecorder = () => (
    <div className="border border-gray-300 rounded-lg p-6">
      <div className="text-center mb-4">
        <Mic className="mx-auto text-gray-400 mb-2" size={32} />
        <h3 className="font-medium text-gray-900">Record Audio</h3>
        <p className="text-xs text-gray-500">Record directly from microphone</p>
      </div>

      {!isRecording && !recordedAudio && (
        <button
          onClick={startRecording}
          disabled={loading}
          className="w-full bg-gradient-to-r from-red-500 to-pink-600 text-white py-3 px-4 rounded-lg font-medium hover:from-red-600 hover:to-pink-700 transition-all flex items-center justify-center space-x-2"
        >
          <Mic size={20} />
          <span>Start Recording</span>
        </button>
      )}

      {isRecording && (
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            <span className="text-red-600 font-medium">Recording... {recordingDuration}s</span>
          </div>
          <button
            onClick={stopRecording}
            className="w-full bg-gradient-to-r from-gray-500 to-gray-600 text-white py-3 px-4 rounded-lg font-medium hover:from-gray-600 hover:to-gray-700 transition-all flex items-center justify-center space-x-2"
          >
            <Square size={20} />
            <span>Stop Recording</span>
          </button>
        </div>
      )}

      {recordedAudio && (
        <div className="space-y-3">
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-green-800">
                Recording ready ({recordingDuration}s)
              </span>
              <button
                onClick={clearRecording}
                className="text-red-600 hover:text-red-800"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </div>
          <audio controls className="w-full">
            <source src={URL.createObjectURL(recordedAudio)} type="audio/wav" />
          </audio>
        </div>
      )}
    </div>
  );

  const MultiFileUpload = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Training Files</h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">
            {retrainFiles.filter(f => f.file && f.label).length} files ready
          </span>
          <button
            onClick={addRetrainFile}
            disabled={loading}
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            <Plus size={16} />
          </button>
        </div>
      </div>

      {retrainFiles.length === 0 && (
        <div className="text-center py-8 border border-dashed border-gray-300 rounded-lg">
          <FileAudio className="mx-auto text-gray-400 mb-2" size={32} />
          <p className="text-gray-500">No files added yet</p>
          <button
            onClick={addRetrainFile}
            className="mt-2 text-blue-600 hover:text-blue-500 font-medium"
          >
            Add your first file
          </button>
        </div>
      )}

      {retrainFiles.map((fileData, index) => (
        <div key={fileData.id} className="border border-gray-200 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-medium text-gray-900">File #{index + 1}</span>
            <button
              onClick={() => removeRetrainFile(fileData.id)}
              className="text-red-600 hover:text-red-800"
            >
              <Minus size={16} />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Audio File</label>
              <div className="border border-dashed border-gray-300 rounded-lg p-3 text-center hover:border-blue-400 transition-colors">
                <label className="cursor-pointer">
                  <span className="text-sm text-blue-600 hover:text-blue-500">
                    {fileData.file ? fileData.file.name : 'Choose file'}
                  </span>
                  <input
                    type="file"
                    accept=".wav"
                    onChange={(e) => updateRetrainFile(fileData.id, 'file', e.target.files[0])}
                    disabled={loading}
                    className="hidden"
                  />
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Label</label>
              <select
                value={fileData.label}
                onChange={(e) => updateRetrainFile(fileData.id, 'label', e.target.value)}
                disabled={loading}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select label</option>
                {modelInfo?.classes?.map(cls => (
                  <option key={cls} value={cls}>{cls}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      ))}

      {retrainFiles.length > 0 && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-blue-900">Batch Training</p>
              <p className="text-sm text-blue-700">
                {canRetrain()
                  ? `Ready to train with ${retrainFiles.filter(f => f.file && f.label).length} files`
                  : `Need ${Math.ceil(retrainFiles.filter(f => f.file && f.label).length / 10) * 10 - retrainFiles.filter(f => f.file && f.label).length} more files (minimum 10, multiples of 10)`
                }
              </p>
            </div>
            <button
              onClick={handleRetrain}
              disabled={!canRetrain() || loading}
              className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-purple-600 hover:to-pink-700 transition-all"
            >
              {loading ? 'Training...' : 'Start Training'}
            </button>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Brain className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">UrbanSound Classifier</h1>
                <p className="text-sm text-gray-500">ML Model Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                <Activity size={12} className="mr-1" />
                Online
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Model Status"
            value="Active"
            icon={Monitor}
            color="text-green-600"
            subtitle="Random Forest"
          />
          <StatCard
            title="Total Classes"
            value={modelInfo?.classes?.length || 0}
            icon={Database}
            color="text-blue-600"
            subtitle="Urban sounds"
          />
          <StatCard
            title="Predictions Made"
            value={predictionHistory.length}
            icon={TrendingUp}
            color="text-purple-600"
            subtitle="All time"
          />
          <StatCard
            title="Training Sessions"
            value={trainingHistory.length}
            icon={Zap}
            color="text-orange-600"
            subtitle="Model updates"
          />
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-2 mb-8 bg-white p-2 rounded-xl border border-gray-200">
          <TabButton
            id="predict"
            label="Predict"
            icon={Play}
            isActive={activeTab === 'predict'}
            onClick={setActiveTab}
          />
          <TabButton
            id="retrain"
            label="Retrain"
            icon={RefreshCw}
            isActive={activeTab === 'retrain'}
            onClick={setActiveTab}
          />
          <TabButton
            id="metrics"
            label="Metrics"
            icon={BarChart3}
            isActive={activeTab === 'metrics'}
            onClick={setActiveTab}
          />
          <TabButton
            id="history"
            label="History"
            icon={History}
            isActive={activeTab === 'history'}
            onClick={setActiveTab}
          />
          <TabButton
            id="settings"
            label="Settings"
            icon={Settings}
            isActive={activeTab === 'settings'}
            onClick={setActiveTab}
          />
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          {/* Predict Tab */}
          {activeTab === 'predict' && (
            <div className="space-y-6">
              <div className="flex items-center space-x-2">
                <Play className="text-blue-600" size={24} />
                <h2 className="text-2xl font-bold text-gray-900">Make Prediction</h2>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <FileUpload
                      file={predictFile && !recordedAudio ? predictFile : null}
                      setFile={setPredictFile}
                      accept=".wav"
                      label="audio file"
                      disabled={loading || isRecording || recordedAudio}
                    />
                    <AudioRecorder />
                  </div>

                  {uploadProgress > 0 && (
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  )}

                  <button
                    onClick={handlePredict}
                    disabled={!predictFile || loading}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-blue-600 hover:to-purple-700 transition-all"
                  >
                    {loading ? 'Processing...' : 'Predict Sound'}
                  </button>
                </div>

                {prediction && (
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-6">
                      <div className="flex items-center space-x-2 mb-3">
                        <CheckCircle className="text-green-600" size={20} />
                        <h3 className="font-semibold text-green-800">Prediction Result</h3>
                      </div>
                      <p className="text-2xl font-bold text-green-800 mb-2">{prediction.prediction}</p>
                      <p className="text-sm text-green-600">Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
                    </div>

                    {prediction.probabilities && (
                      <div className="space-y-2">
                        <h4 className="font-medium text-gray-900">All Probabilities</h4>
                        {Object.entries(prediction.probabilities).map(([className, data]) => (
                          <div key={className} className="flex items-center justify-between p-2 rounded bg-gray-50">
                            <span className="text-sm font-medium">{className}</span>
                            <div className="flex items-center space-x-2">
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${data.is_predicted ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-gray-300'}`}
                                  style={{ width: `${data.probability * 100}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-600 w-12">{(data.probability * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Retrain Tab */}
          {activeTab === 'retrain' && (
            <div className="space-y-6">
              <div className="flex items-center space-x-2">
                <RefreshCw className="text-purple-600" size={24} />
                <h2 className="text-2xl font-bold text-gray-900">Retrain Model</h2>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2">
                  <MultiFileUpload />
                </div>

                <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg p-6">
                  <div className="flex items-center space-x-2 mb-3">
                    <AlertCircle className="text-yellow-600" size={20} />
                    <h3 className="font-semibold text-yellow-800">Retraining Info</h3>
                  </div>
                  <ul className="text-sm text-yellow-700 space-y-2">
                    <li>• Upload files in batches of 10, 20, 30, etc.</li>
                    <li>• Each file needs a proper label</li>
                    <li>• Model will be updated incrementally</li>
                    <li>• Previous performance will be saved</li>
                    <li>• Process may take several minutes</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="text-green-600" size={24} />
                  <h2 className="text-2xl font-bold text-gray-900">Model Metrics</h2>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={fetchMetrics}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                  >
                    Calculate Metrics
                  </button>
                  <button
                    onClick={resetMetrics}
                    className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {/* Log Prediction Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Log Prediction for Metrics</h3>
                  <FileUpload
                    file={logFile}
                    setFile={setLogFile}
                    accept=".wav"
                    label="test audio"
                    disabled={loading}
                  />

                  <select
                    value={logLabel}
                    onChange={(e) => setLogLabel(e.target.value)}
                    disabled={loading}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select true label</option>
                    {modelInfo?.classes?.map(cls => (
                      <option key={cls} value={cls}>{cls}</option>
                    ))}
                  </select>

                  <button
                    onClick={handleLogPrediction}
                    disabled={!logFile || !logLabel || loading}
                    className="w-full bg-gradient-to-r from-green-500 to-teal-600 text-white py-3 px-4 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-green-600 hover:to-teal-700 transition-all"
                  >
                    {loading ? 'Logging...' : 'Log Prediction'}
                  </button>
                </div>

                {metrics && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
                        <p className="text-sm text-blue-600 font-medium">Accuracy</p>
                        <p className="text-2xl font-bold text-blue-800">{(metrics.accuracy * 100).toFixed(1)}%</p>
                      </div>
                      <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
                        <p className="text-sm text-green-600 font-medium">Samples</p>
                        <p className="text-2xl font-bold text-green-800">{metrics.total_samples}</p>
                      </div>
                    </div>

                    {metrics.classification_report?.weighted_avg && (
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
                          <p className="text-sm text-purple-600 font-medium">Precision</p>
                          <p className="text-xl font-bold text-purple-800">
                            {(metrics.classification_report.weighted_avg.precision * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="bg-gradient-to-r from-orange-50 to-orange-100 p-4 rounded-lg">
                          <p className="text-sm text-orange-600 font-medium">Recall</p>
                          <p className="text-xl font-bold text-orange-800">
                            {(metrics.classification_report.weighted_avg.recall * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="bg-gradient-to-r from-pink-50 to-pink-100 p-4 rounded-lg">
                          <p className="text-sm text-pink-600 font-medium">F1-Score</p>
                          <p className="text-xl font-bold text-pink-800">
                            {(metrics.classification_report.weighted_avg['f1-score'] * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <div className="space-y-6">
              <div className="flex items-center space-x-2">
                <History className="text-orange-600" size={24} />
                <h2 className="text-2xl font-bold text-gray-900">History</h2>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Training History */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Training History</h3>
                  <div className="space-y-3">
                    {trainingHistory.map((entry, index) => (
                      <div key={entry.id} className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-4">
                        <span className="font-bold mb-5 block text-green-500">({entry.model_name})</span>
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">Training #{trainingHistory.length - index}</span>
                          <span className="text-sm text-gray-500">
                            {new Date(entry.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">Accuracy:</span>
                            <span className="font-medium ml-1">{(entry.accuracy * 100).toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Precision:</span>
                            <span className="font-medium ml-1">{(entry.precision * 100).toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-gray-600">recall:</span>
                            <span className="font-medium ml-1">{(entry.recall * 100).toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-gray-600">F1-Score:</span>
                            <span className="font-medium ml-1">{(entry.f1_score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        {entry.is_retraining && (
                          <div className="mt-2 flex items-center space-x-1">
                            <RefreshCw size={12} className="text-purple-600" />
                            <span className="text-xs text-purple-600">Retraining Session</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Prediction History */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Predictions</h3>
                  <div className="space-y-3">
                    {predictionHistory.slice(0, 5).map((entry) => (
                      <div key={entry.id} className="bg-gradient-to-r from-green-50 to-teal-50 border border-green-200 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900 truncate">{entry.file_name}</span>
                          {entry.is_correct !== null && (
                            entry.is_correct ?
                              <CheckCircle className="text-green-600" size={16} /> :
                              <XCircle className="text-red-600" size={16} />
                          )}
                        </div>
                        <div className="text-sm space-y-1">
                          <div>
                            <span className="text-gray-600">Predicted:</span>
                            <span className="font-medium ml-1">{entry.predicted_label}</span>
                          </div>
                          {entry.true_label && (
                            <div>
                              <span className="text-gray-600">Actual:</span>
                              <span className="font-medium ml-1">{entry.true_label}</span>
                            </div>
                          )}
                          {entry.confidence && (
                            <div>
                              <span className="text-gray-600">Confidence:</span>
                              <span className="font-medium ml-1">{(entry.confidence * 100).toFixed(1)}%</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Settings Tab */}
          {activeTab === 'settings' && (
            <div className="space-y-6">
              <div className="flex items-center space-x-2">
                <Settings className="text-gray-600" size={24} />
                <h2 className="text-2xl font-bold text-gray-900">Model Information</h2>
              </div>

              {modelInfo && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200 rounded-lg p-6">
                      <h3 className="font-semibold text-gray-900 mb-4">Model Details</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Type:</span>
                          <span className="font-medium">{modelInfo.model_type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Classes:</span>
                          <span className="font-medium">{modelInfo.classes.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Status:</span>
                          <span className="font-medium text-green-600">Active</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-6">
                      <h3 className="font-semibold text-blue-900 mb-4">Metrics Tracking</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-blue-700">Labels Count:</span>
                          <span className="font-medium">{modelInfo.metrics_tracking.labels_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-blue-700">Predictions Count:</span>
                          <span className="font-medium">{modelInfo.metrics_tracking.predictions_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-blue-700">Balanced:</span>
                          <span className={`font-medium ${modelInfo.metrics_tracking.balanced ? 'text-green-600' : 'text-red-600'}`}>
                            {modelInfo.metrics_tracking.balanced ? 'Yes' : 'No'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="bg-gradient-to-r from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-6">
                      <h3 className="font-semibold text-purple-900 mb-4">Available Classes</h3>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {modelInfo.classes.map((cls, index) => (
                          <div key={cls} className="bg-white px-3 py-2 rounded border">
                            <span className="font-medium text-purple-800">{cls}</span>
                            <span className="text-xs text-gray-500 ml-2">#{index}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-lg p-6">
                <h3 className="font-semibold text-yellow-900 mb-4">API Endpoints</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center justify-between p-3 bg-white rounded border">
                    <span className="font-mono text-blue-600">POST /predict</span>
                    <span className="text-gray-600">Make predictions</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-white rounded border">
                    <span className="font-mono text-purple-600">POST /retrain</span>
                    <span className="text-gray-600">Retrain model</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-white rounded border">
                    <span className="font-mono text-green-600">GET /metrics</span>
                    <span className="text-gray-600">Get performance metrics</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-white rounded border">
                    <span className="font-mono text-orange-600">GET /model_info</span>
                    <span className="text-gray-600">Model information</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;