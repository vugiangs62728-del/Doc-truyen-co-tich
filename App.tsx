import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, Blob } from '@google/genai';
import { ConnectionState, Transcript } from './types';

// --- Audio Utility Functions ---

function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): Blob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        int16[i] = data[i] < 0 ? data[i] * 32768 : data[i] * 32767;
    }
    return {
        data: encode(new Uint8Array(int16.buffer)),
        mimeType: 'audio/pcm;rate=16000',
    };
}


// --- UI Components (defined outside App to prevent re-creation) ---

interface AudioVisualizerProps {
  analyserNode: AnalyserNode | null;
  isActive: boolean;
}
const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ analyserNode, isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!analyserNode || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;

    analyserNode.fftSize = 256;
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let animationFrameId: number;

    const draw = () => {
      animationFrameId = requestAnimationFrame(draw);
      analyserNode.getByteTimeDomainData(dataArray);

      canvasCtx.fillStyle = 'rgb(17 24 39)'; // bg-gray-900
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      canvasCtx.lineWidth = 2;
      
      const gradient = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
      gradient.addColorStop(0, '#3b82f6'); // blue-500
      gradient.addColorStop(0.5, '#9333ea'); // purple-600
      gradient.addColorStop(1, '#ec4899'); // pink-500
      canvasCtx.strokeStyle = isActive ? gradient : 'rgb(75 85 99)'; // gray-500

      canvasCtx.beginPath();
      const sliceWidth = canvas.width * 1.0 / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    };
    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [analyserNode, isActive]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
};

const StatusIndicator: React.FC<{ state: ConnectionState }> = ({ state }) => {
    const statusMap = {
        [ConnectionState.IDLE]: { text: 'Ready', color: 'bg-gray-500' },
        [ConnectionState.CONNECTING]: { text: 'Connecting...', color: 'bg-yellow-500 animate-pulse' },
        [ConnectionState.CONNECTED]: { text: 'Listening', color: 'bg-green-500' },
        [ConnectionState.DISCONNECTED]: { text: 'Disconnected', color: 'bg-red-500' },
        [ConnectionState.ERROR]: { text: 'Error', color: 'bg-red-700' },
    };
    const { text, color } = statusMap[state];
    return (
        <div className="absolute top-4 right-4 flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${color}`}></div>
            <span className="text-sm text-gray-300">{text}</span>
        </div>
    );
};

const TranscriptionLog: React.FC<{ transcripts: Transcript[] }> = ({ transcripts }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if(scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [transcripts]);

    return (
        <div ref={scrollRef} className="w-full max-w-4xl flex-grow space-y-4 overflow-y-auto p-4">
            {transcripts.map((t) => (
                <div key={t.id} className={`flex ${t.speaker === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-prose p-3 rounded-lg ${t.speaker === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-200'}`}>
                       <p className={`opacity-${t.isFinal ? '100' : '70'}`}>{t.text}</p>
                    </div>
                </div>
            ))}
        </div>
    );
};

// --- Main App Component ---

const App: React.FC = () => {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.IDLE);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  
  const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const outputSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const inputAnalyserNodeRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserNodeRef = useRef<AnalyserNode | null>(null);

  const nextStartTimeRef = useRef<number>(0);
  const transcriptIdCounterRef = useRef<number>(0);

  const systemInstruction = `You are a storyteller for young children. Speak with a gentle, warm, and soft voice, full of wonder and kindness, like a little girl telling her favorite bedtime story. Your tone should be innocent and emotional. Use a clear American accent and speak at a slightly slow, melodic pace. Pause softly between sentences and emphasize magical or emotional words. Express wonder, kindness, and other gentle emotions naturally in your storytelling.`;
  
  const cleanup = useCallback(() => {
    sessionPromiseRef.current?.then(session => session.close()).catch(() => {});
    
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    scriptProcessorRef.current?.disconnect();

    inputAudioContextRef.current?.close().catch(() => {});
    outputAudioContextRef.current?.close().catch(() => {});
    
    outputSourcesRef.current.forEach(source => source.stop());

    sessionPromiseRef.current = null;
    mediaStreamRef.current = null;
    inputAudioContextRef.current = null;
    outputAudioContextRef.current = null;
    scriptProcessorRef.current = null;
    outputSourcesRef.current.clear();
    inputAnalyserNodeRef.current = null;
    outputAnalyserNodeRef.current = null;
    nextStartTimeRef.current = 0;

    setConnectionState(ConnectionState.IDLE);
  }, []);

  const handleToggleListening = async () => {
    if (connectionState !== ConnectionState.IDLE) {
      cleanup();
      return;
    }

    try {
      setTranscripts([]);
      setConnectionState(ConnectionState.CONNECTING);
      
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // Fix: Use a cross-browser compatible AudioContext, casting window to any to avoid TypeScript errors for webkitAudioContext.
      const AudioContext = (window as any).AudioContext || (window as any).webkitAudioContext;
      const inputCtx = new AudioContext({ sampleRate: 16000 });
      inputAudioContextRef.current = inputCtx;
      const outputCtx = new AudioContext({ sampleRate: 24000 });
      outputAudioContextRef.current = outputCtx;

      const micSource = inputCtx.createMediaStreamSource(stream);
      const processor = inputCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;
      
      const inputAnalyser = inputCtx.createAnalyser();
      inputAnalyserNodeRef.current = inputAnalyser;

      const outputGain = outputCtx.createGain();
      const outputAnalyser = outputCtx.createAnalyser();
      outputAnalyserNodeRef.current = outputAnalyser;
      outputGain.connect(outputAnalyser);
      outputAnalyser.connect(outputCtx.destination);
      
      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
          systemInstruction,
        },
        callbacks: {
          onopen: () => setConnectionState(ConnectionState.CONNECTED),
          onclose: () => cleanup(),
          onerror: (e) => {
            console.error("Gemini API Error:", e);
            setConnectionState(ConnectionState.ERROR);
            cleanup();
          },
          onmessage: (msg: LiveServerMessage) => handleServerMessage(msg, outputCtx, outputGain),
        },
      });

      processor.onaudioprocess = (audioProcessingEvent) => {
        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
        const pcmBlob = createBlob(inputData);
        sessionPromiseRef.current?.then((session) => {
          session.sendRealtimeInput({ media: pcmBlob });
        });
      };
      
      micSource.connect(inputAnalyser);
      inputAnalyser.connect(processor);
      processor.connect(inputCtx.destination);
      
    } catch (error) {
      console.error("Failed to start listening:", error);
      setConnectionState(ConnectionState.ERROR);
      cleanup();
    }
  };

  const handleServerMessage = async (msg: LiveServerMessage, outputCtx: AudioContext, outputGain: GainNode) => {
      if (msg.serverContent?.inputTranscription) {
          const { text, isFinal } = msg.serverContent.inputTranscription;
          updateTranscript('user', text, isFinal);
      }
      if (msg.serverContent?.outputTranscription) {
          const { text, isFinal } = msg.serverContent.outputTranscription;
          updateTranscript('model', text, isFinal);
      }

      if (msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data) {
          const base64Audio = msg.serverContent.modelTurn.parts[0].inlineData.data;
          const audioBuffer = await decodeAudioData(decode(base64Audio), outputCtx, 24000, 1);
          
          nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
          
          const source = outputCtx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(outputGain);
          source.start(nextStartTimeRef.current);
          
          nextStartTimeRef.current += audioBuffer.duration;
          
          outputSourcesRef.current.add(source);
          source.onended = () => outputSourcesRef.current.delete(source);
      }

      if (msg.serverContent?.interrupted) {
          outputSourcesRef.current.forEach(source => source.stop());
          outputSourcesRef.current.clear();
          nextStartTimeRef.current = 0;
      }
  };

  const updateTranscript = (speaker: 'user' | 'model', textChunk: string, isFinalChunk: boolean) => {
    setTranscripts(prev => {
      const last = prev[prev.length - 1];
      if (last && last.speaker === speaker && !last.isFinal) {
        const updated = [...prev];
        updated[prev.length - 1] = {
          ...last,
          text: last.text + textChunk,
          isFinal: isFinalChunk,
        };
        return updated;
      } else {
        return [
          ...prev,
          {
            id: transcriptIdCounterRef.current++,
            speaker,
            text: textChunk,
            isFinal: isFinalChunk,
          },
        ];
      }
    });
  };

  useEffect(() => {
    return () => cleanup();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isListening = connectionState === ConnectionState.CONNECTING || connectionState === ConnectionState.CONNECTED;

  return (
    <div className="bg-gray-900 text-white h-screen w-screen flex flex-col items-center justify-center p-4 font-sans relative">
      <div className="absolute top-4 left-4">
        <h1 className="text-2xl font-bold text-gray-200">Gemini Storyteller</h1>
        <p className="text-sm text-gray-400">Talk to a friendly storyteller</p>
      </div>
      <StatusIndicator state={connectionState} />

      <TranscriptionLog transcripts={transcripts} />

      <div className="w-full max-w-4xl grid grid-cols-2 gap-4 h-20 my-4">
        <div className="bg-gray-800 rounded-lg p-2 flex flex-col items-center justify-center">
            <p className="text-xs text-gray-400 mb-1">Your Voice</p>
            <AudioVisualizer analyserNode={inputAnalyserNodeRef.current} isActive={isListening} />
        </div>
        <div className="bg-gray-800 rounded-lg p-2 flex flex-col items-center justify-center">
            <p className="text-xs text-gray-400 mb-1">Storyteller's Voice</p>
            <AudioVisualizer analyserNode={outputAnalyserNodeRef.current} isActive={isListening}/>
        </div>
      </div>
      
      <div className="flex-shrink-0 p-4">
        <button
          onClick={handleToggleListening}
          className={`relative w-20 h-20 rounded-full transition-all duration-300 ease-in-out flex items-center justify-center shadow-lg focus:outline-none focus:ring-4 focus:ring-opacity-50
            ${isListening ? 'bg-red-600 hover:bg-red-700 focus:ring-red-400' : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-400'}`}
        >
          {isListening && <span className="absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-75 animate-ping"></span>}
          <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default App;
