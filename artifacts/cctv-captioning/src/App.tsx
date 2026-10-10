import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  Aperture,
  ArrowUpFromLine,
  Check,
  Clipboard,
  Cpu,
  FileVideo,
  Film,
  HardDrive,
  LoaderCircle,
  LockKeyhole,
  MonitorPlay,
  RefreshCw,
  ScanEye,
  ShieldCheck,
  Sparkles,
  Trash2,
  TriangleAlert,
  WifiOff,
} from 'lucide-react';
import {
  getGetCaptioningHealthQueryKey,
  useAnalyzeVideo,
  useGetCaptioningHealth,
} from '@workspace/api-client-react';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import type { ObjectEvidence, VideoAnalysis } from '@workspace/api-client-react';

const queryClient = new QueryClient();

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(value?: number | null) {
  if (value == null || !Number.isFinite(value)) return null;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return minutes ? `${minutes}m ${seconds.toString().padStart(2, '0')}s` : `${seconds}s`;
}

function errorMessage(error: unknown) {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string') return error;
  return 'The video could not be analyzed. Please check the file and try again.';
}

function EvidenceRow({ evidence }: { evidence: ObjectEvidence }) {
  return (
    <div className="evidence-item" data-testid={`evidence-row-${evidence.label}`}>
      <span className="evidence-label">{evidence.label}</span>
      <span className="evidence-explanation">{evidence.interpretation}</span>
      <span className="evidence-count">
        {evidence.detection_hits_across_frames} hits · {evidence.sampled_frames_present}/{evidence.sampled_frame_count} frames · max {evidence.max_confidence}
      </span>
    </div>
  );
}

function AnalysisResult({ result }: { result: VideoAnalysis }) {
  const [copied, setCopied] = useState(false);

  const copyCaption = async () => {
    try {
      await navigator.clipboard.writeText(result.caption);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="result-content" data-testid="result-analysis">
      <div className="result-banner">
        <div className="result-file" data-testid="text-result-filename">
          <FileVideo size={15} />
          <span>{result.video_filename}</span>
        </div>
      </div>

      <div className="caption-label">
        <span>Generated caption</span>
        <button className="copy-button" type="button" onClick={copyCaption} data-testid="button-copy-caption">
          {copied ? <Check size={12} /> : <Clipboard size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <p className="caption-text" data-testid="text-caption">{result.caption}</p>
    </div>
  );
}

function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<VideoAnalysis | null>(null);
  const [localError, setLocalError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const analysis = useAnalyzeVideo();
  const healthQuery = useGetCaptioningHealth({
    query: {
      queryKey: getGetCaptioningHealthQueryKey(),
      refetchInterval: 30_000,
      refetchOnWindowFocus: true,
    },
  });
  const busy = analysis.isPending;
  const previewUrl = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const acceptFile = (candidate?: File) => {
    if (!candidate) return;
    const extension = candidate.name.split('.').pop()?.toLowerCase();
    if (!extension || !['mp4', 'avi', 'mov', 'mkv'].includes(extension)) {
      setLocalError('Choose an MP4, AVI, MOV, or MKV video file.');
      return;
    }
    setLocalError('');
    setResult(null);
    setFile(candidate);
  };

  const removeFile = () => {
    if (busy) return;
    setFile(null);
    setResult(null);
    setLocalError('');
    if (fileInput.current) fileInput.current.value = '';
  };

  const submit = () => {
    if (!file || busy) return;
    setLocalError('');
    setResult(null);
    analysis.mutate({ data: { video: file } }, {
      onSuccess: (response) => {
        if (response.success) {
          setResult(response);
        } else {
          setLocalError('The service did not return a successful analysis. Try another video or retry this file.');
        }
      },
      onError: (error) => setLocalError(errorMessage(error)),
    });
  };

  const health = healthQuery.data;
  const healthIsReady = Boolean(health?.model_loaded);
  const healthIsOffline = healthQuery.isError || (health !== undefined && health.status !== 'ok');

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Aperture size={19} strokeWidth={1.8} /></div>
          <div>
            <div className="brand-name">Fieldnote / Vision</div>
            <div className="brand-sub">Video captioning workspace</div>
          </div>
        </div>
        <div className="topbar-note"><ShieldCheck size={16} /><span>Factual descriptions. No added labels.</span></div>
      </header>

      <div className="workspace">
        <section className="page-intro">
          <div>
            <div className="eyebrow">CCTV review tool</div>
            <h1>Make footage easier<br />to review.</h1>
            <p>Upload one video to receive a concise, factual caption from the Qwen3-VL model, with processing details for review.</p>
          </div>
          <div className="workflow-label">01 <span>Upload</span> — 02 <span>Inspect caption</span></div>
        </section>

        <section className="columns" aria-label="Video captioning workflow">
          <section className="panel" aria-labelledby="upload-heading">
            <div className="panel-head">
              <div className="panel-title" id="upload-heading"><span className="step-num">01</span> Video source</div>
              <span className="panel-kicker">One file</span>
            </div>
            <div className="panel-body">
              <div
                className={`dropzone${dragActive ? ' drag-active' : ''}`}
                onDragEnter={(event) => { event.preventDefault(); if (!busy) setDragActive(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => { event.preventDefault(); setDragActive(false); }}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                  if (!busy) acceptFile(event.dataTransfer.files[0]);
                }}
                data-testid="dropzone-video"
              >
                <div className="drop-content">
                  <div className="upload-icon"><ArrowUpFromLine size={21} /></div>
                  <div className="drop-title">{file ? 'Replace selected video' : 'Drop a video file here'}</div>
                  <div className="drop-subtitle">or choose a file from your device</div>
                  <input
                    ref={fileInput}
                    className="file-input"
                    type="file"
                    accept=".mp4,.avi,.mov,.mkv"
                    disabled={busy}
                    onChange={(event) => acceptFile(event.target.files?.[0])}
                    data-testid="input-video-file"
                    aria-label="Choose a CCTV video file"
                  />
                  <button className="choose-button" type="button" disabled={busy} onClick={() => fileInput.current?.click()} data-testid="button-select-video">
                    Browse files
                  </button>
                  <div className="format-note">MP4, AVI, MOV or MKV</div>
                </div>
              </div>

              {file && (
                <div className="file-card" data-testid="card-selected-file">
                  <div className="file-glyph"><Film size={18} /></div>
                  <div className="file-details">
                    <div className="file-name" title={file.name}>{file.name}</div>
                    <div className="file-size">{formatBytes(file.size)} <span aria-hidden="true">·</span> Local preview</div>
                  </div>
                  <button className="icon-button" type="button" aria-label="Remove selected video" disabled={busy} onClick={removeFile} data-testid="button-remove-video">
                    <Trash2 size={15} />
                  </button>
                </div>
              )}

              {file && previewUrl && (
                <div className="video-frame">
                  <video src={previewUrl} controls playsInline preload="metadata" aria-label={`Preview of ${file.name}`} data-testid="video-preview" />
                </div>
              )}

              <div className="privacy-note"><LockKeyhole size={14} /><span>Your selected video is submitted to the captioning service. The preview is generated locally in this browser.</span></div>
              <button className="analyze-button" type="button" onClick={submit} disabled={!file || busy} data-testid="button-analyze-video">
                {busy ? <><LoaderCircle className="animate-spin" size={16} /> Analyzing video</> : <><ScanEye size={16} /> Analyze video <span aria-hidden="true">→</span></>}
              </button>
              {busy && <div className="loading-line" role="status" aria-label="Analysis in progress" data-testid="status-analysis-loading" />}
              {localError && (
                <div className="error-box" role="alert" data-testid="status-analysis-error">
                  <TriangleAlert size={16} />
                  <div><strong>Analysis could not be completed</strong>{localError}</div>
                </div>
              )}
            </div>
          </section>

          <section className="panel result-panel" aria-labelledby="result-heading">
            <div className="panel-head">
              <div className="panel-title" id="result-heading"><span className="step-num">02</span> Caption &amp; evidence</div>
              <span className="panel-kicker">Review output</span>
            </div>
            {result ? (
              <AnalysisResult result={result} />
            ) : busy ? (
              <div className="empty-result" role="status" data-testid="status-result-loading">
                <div className="empty-illustration"><Cpu size={29} strokeWidth={1.4} /></div>
                <h2>Reviewing the footage</h2>
                <p>Qwen3-VL is processing the selected video. This can take a little while depending on its length.</p>
                <div className="empty-index">Analysis in progress</div>
              </div>
            ) : (
              <div className="empty-result" data-testid="empty-analysis-result">
                <div className="empty-illustration"><MonitorPlay size={30} strokeWidth={1.4} /></div>
                <h2>Your result will appear here</h2>
                <p>Choose a CCTV video and run an analysis to inspect its caption and supporting processing metadata.</p>
                <div className="empty-index"><Sparkles size={12} /> Awaiting source video</div>
              </div>
            )}
          </section>
        </section>
      </div>

      <footer className="footer">
        <span className="footer-mark"><HardDrive size={12} /> Captioning service</span>
        <span>Qwen3-VL · Review aid, not a determination</span>
      </footer>
    </main>
  );
}

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
