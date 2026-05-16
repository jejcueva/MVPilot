"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type StepStatus = "Ready" | "Running" | "Pending" | "Complete" | "Failed";
type AdditionalSourceType = "url" | "file";
type RepoPreference = "create_new_repo" | "use_existing_repo";
type RepoVisibility = "private" | "public";
type FlightStageKey = "preflight" | "radar_scan" | "flight_plan" | "autopilot" | "black_box" | "landed";
type GithubStatus = "not_connected" | "connecting" | "connected" | "error";

type AgentStep = {
  key: FlightStageKey;
  phase: string;
  title: string;
  detail: string;
  status: StepStatus;
};

type WorkflowStep = {
  node_name: string;
  status: string;
  message: string;
  flight_stage?: string | null;
  agent?: string | null;
  timestamp?: string;
};

type RagEvidence = {
  source: string;
  docType?: string;
  doc_type?: string;
  chunkId?: string;
  chunk_id?: string;
  content?: string;
  text?: string;
  score?: number;
};

type ToolCall = {
  tool?: string;
  status?: string;
  summary?: string;
  commit_url?: string | null;
  repo?: unknown;
  files?: unknown;
};

type GeneratedArtifact = {
  name: string;
  kind?: string;
  summary?: string;
  content?: string;
};

type BuildTimelineEvent = {
  id: string;
  title: string;
  category?: string;
  status: string;
  detail?: string;
  artifacts?: string[];
  updated_at?: string | null;
};

type MvpPlan = {
  title?: string | null;
  idea?: string;
  target_users?: string | null;
  tech_stack_preference?: string | null;
  features?: string[];
  architecture_notes?: string | null;
  implementation_steps?: string[];
  selected_stack?: string | null;
  runtime?: string;
};

type TaskDetail = {
  task: {
    id: string;
    status: string;
    repo_visibility: RepoVisibility;
    repo_description?: string | null;
  };
  runtime?: string;
  mvp_plan?: MvpPlan | null;
  build_timeline?: BuildTimelineEvent[];
  mvp_validation?: {
    passed?: boolean;
    project_title?: string;
    checks?: { name: string; passed: boolean; detail: string }[];
  } | null;
  mvp_delivery?: {
    project_title?: string;
    completed_features?: string[];
    mocked_features?: string[];
    pending_features?: string[];
    validation_passed?: boolean;
    model_modes?: string[];
  } | null;
  agent_steps: WorkflowStep[];
  build_context?: {
    evidence?: RagEvidence[];
    requiredDeliverables?: unknown[];
    scopeWarnings?: unknown[];
  };
  tool_calls?: ToolCall[];
  generated_artifacts?: GeneratedArtifact[];
  final_report?: {
    status?: string;
    summary?: string;
    mvp_plan?: TaskDetail["mvp_plan"];
    build_timeline?: TaskDetail["build_timeline"];
    mvp_delivery?: TaskDetail["mvp_delivery"];
    mvp_validation?: TaskDetail["mvp_validation"];
    repo?: { url?: string | null; name?: string | null } | null;
    links?: {
      repoUrl?: string | null;
      commitUrl?: string | null;
      branch?: string | null;
      buildLogPath?: string | null;
      architectureDocPath?: string | null;
      demoScriptPath?: string | null;
    };
  } | null;
};

type AdditionalSource =
  | { id: number; type: "url"; value: string }
  | { id: number; type: "file"; file: File | null };

const defaultTitle = "StudyPilot";
const defaultIdea =
  "Build StudyPilot, an AI study planner that turns messy course goals into weekly plans, focus sessions, and progress dashboards for college students.";
/** Optional product/rules URL prefilled for localhost demo; clear the field to skip RAG fetch. */
const defaultReferenceUrl = "https://www.shortesthack.com/?tab=rules";

const flightStops: AgentStep[] = [
  { key: "preflight", phase: "PREFLIGHT", title: "Preflight Check", detail: "Package the idea, repo preference, GitHub connection, rules URL, and extra source material.", status: "Ready" },
  { key: "radar_scan", phase: "RADAR", title: "Radar Scan", detail: "Retrieve hackathon rules, NVIDIA docs, uploaded files, logs, and RAG evidence.", status: "Pending" },
  { key: "flight_plan", phase: "PLAN", title: "Flight Plan", detail: "Nemotron uses RAG build context to generate the implementation plan.", status: "Pending" },
  { key: "autopilot", phase: "AUTOPILOT", title: "Autopilot Engaged", detail: "GitHub Agent creates or updates the repo, commits files, and verifies outputs.", status: "Pending" },
  { key: "black_box", phase: "BLACK BOX", title: "Black Box Recorder", detail: "Store logs, decisions, artifacts, errors, and final memory.", status: "Pending" },
  { key: "landed", phase: "LANDED", title: "MVP Landed", detail: "Final repo, commit, build log, and architecture links are ready.", status: "Pending" },
];

const sourceTypes = [
  "Optional product, rules, or API documentation URL",
  "Additional OpenClaw or Nemotron documentation URLs",
  "Optional uploaded README, TXT, Markdown, JSON, or CSV files",
];

type GithubOAuthConfig = {
  oauthConfigured: boolean;
  redirectUri?: string;
  missingEnv?: string[];
};

function clearGithubConnectingFlags() {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem("mvpilot_github_connecting");
  window.sessionStorage.removeItem("mvpilot_github_connecting_at");
}

function markGithubConnecting() {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem("mvpilot_github_connecting", "true");
  window.sessionStorage.setItem("mvpilot_github_connecting_at", String(Date.now()));
}

function githubConnectingExpired(): boolean {
  if (typeof window === "undefined") return false;
  const startedAt = Number(window.sessionStorage.getItem("mvpilot_github_connecting_at") || "0");
  if (!startedAt) return false;
  return Date.now() - startedAt > 120_000;
}

function readGithubCallbackState(): {
  connectionId: string | null;
  username: string | null;
  error: string | null;
} {
  if (typeof window === "undefined") {
    return { connectionId: null, username: null, error: null };
  }

  const params = new URLSearchParams(window.location.search);
  const connectionId = params.get("github_connection_id");
  const status = params.get("github_status");
  const error = params.get("github_error");
  const username = params.get("github_username");

  if (status === "error") {
    return { connectionId: null, username: null, error: error || "GitHub connection failed." };
  }

  if (connectionId && (status === "connected" || status === "ready")) {
    window.sessionStorage.setItem("mvpilot_github_connection_id", connectionId);
    if (username) window.sessionStorage.setItem("mvpilot_github_username", username);
    clearGithubConnectingFlags();
    return { connectionId, username, error: null };
  }

  const stored = window.sessionStorage.getItem("mvpilot_github_connection_id");
  const storedUsername = window.sessionStorage.getItem("mvpilot_github_username");
  return { connectionId: stored, username: storedUsername, error: null };
}

function githubReturnTo() {
  if (typeof window === "undefined") return "http://localhost:3000";
  return `${window.location.origin}${window.location.pathname}`;
}

function formatWorkflowError(message: string): string {
  if (message.includes("Resource not accessible by personal access token")) {
    return [
      "GitHub rejected repo creation for the connected OAuth session.",
      "Reconnect GitHub and confirm the OAuth app requested the repo scope.",
      "You can also choose an existing repository that your account can write to.",
    ].join(" ");
  }
  if (message.includes("already exists") || message.includes("name already exists")) {
    return [
      message,
      "MVPilot will reuse an existing repo with the same name on the next launch, or you can change the repo name below.",
    ].join(" ");
  }
  if (message.includes("Git Repository is empty") || message.includes("repository is empty")) {
    return [
      "The existing GitHub repo has no commits yet.",
      "MVPilot will now seed an initial commit automatically — click New launch and try again.",
      "Or add any file (e.g. README) on github.com first, then relaunch.",
    ].join(" ");
  }
  return message;
}

async function fetchGithubStatus(apiBaseUrl: string, connectionId: string | null) {
  if (!connectionId) return { connected: false, username: null, status: "missing" };
  const response = await fetch(`${apiBaseUrl}/api/auth/github/status?github_connection_id=${encodeURIComponent(connectionId)}`, {
    cache: "no-store",
  });
  if (!response.ok) return { connected: false, username: null, status: "error" };
  return await response.json() as { connected: boolean; username?: string | null; status?: string };
}

function githubStatusPill(status: GithubStatus): StepStatus {
  if (status === "connected") return "Complete";
  if (status === "connecting") return "Running";
  if (status === "error") return "Failed";
  return "Ready";
}

function githubStatusLabel(status: GithubStatus): string {
  if (status === "connected") return "Connected";
  if (status === "connecting") return "Connecting";
  if (status === "error") return "Error";
  return "Not connected";
}

function StatusPill({ status }: { status: string }) {
  const styles: Record<string, string> = {
    Ready: "border-[#3a494b] bg-[#1c1f29] text-[#b9cacb]",
    Running: "border-[#00f2ff]/40 bg-[#00f2ff]/10 text-[#00f2ff] shadow-[0_0_12px_rgba(0,242,255,0.18)]",
    Pending: "border-[#3a494b] bg-[#181b25] text-[#849495]",
    Complete: "border-[#4edea3]/40 bg-[#4edea3]/10 text-[#4edea3] shadow-[0_0_12px_rgba(78,222,163,0.18)]",
    Failed: "border-[#ffb4ab]/50 bg-[#93000a]/40 text-[#ffb4ab]",
  };

  return (
    <span className={`rounded-full border px-2.5 py-1 font-mono text-[11px] font-bold uppercase tracking-widest ${styles[status] ?? styles.Pending}`}>
      {status}
    </span>
  );
}

function PlaneMarker({ progress }: { progress: number }) {
  return (
    <div className="absolute top-0 z-20 -translate-x-1/2 transition-all duration-700" style={{ left: `${progress}%` }}>
      <div className="flex items-center gap-1 rounded-full border border-[#00f2ff]/50 bg-[#00f2ff] px-2 py-1 text-[10px] font-black tracking-widest text-[#00363a] shadow-[0_0_18px_rgba(0,242,255,0.55)]">
        <span className="h-0 w-0 border-y-[5px] border-l-[9px] border-y-transparent border-l-[#00363a]" />
        MVP
      </div>
    </div>
  );
}

function deriveFlightStageState(taskDetail: TaskDetail | null, hasLaunched: boolean, hasError: boolean) {
  const completedStages = new Set<string>();
  let failedStage: string | null = null;

  for (const step of taskDetail?.agent_steps ?? []) {
    const stage = step.flight_stage;
    if (!stage) continue;
    if (step.status === "failed" || stage === "failed") {
      failedStage = stage === "failed" ? failedStage : stage;
      continue;
    }
    if (["completed", "success", "blocked"].includes(step.status)) {
      completedStages.add(stage);
    }
  }

  const firstIncompleteIndex = flightStops.findIndex((stop) => !completedStages.has(stop.key));
  const activeStopIndex = hasError
    ? Math.max(0, flightStops.findIndex((stop) => stop.key === failedStage))
    : firstIncompleteIndex === -1
      ? flightStops.length - 1
      : Math.max(0, firstIncompleteIndex);

  const steps = flightStops.map((step, index) => {
    if (hasError && (step.key === failedStage || index === activeStopIndex)) return { ...step, status: "Failed" as StepStatus };
    if (!hasLaunched && index === 0) return { ...step, status: "Ready" as StepStatus };
    if (completedStages.has(step.key)) return { ...step, status: "Complete" as StepStatus };
    if (hasLaunched && index === activeStopIndex) return { ...step, status: "Running" as StepStatus };
    return { ...step, status: "Pending" as StepStatus };
  });

  const completeCount = steps.filter((step) => step.status === "Complete").length;
  const progressPercent = Math.min(100, Math.round((completeCount / (flightStops.length - 1)) * 100));
  return { activeStopIndex, progressPercent: hasLaunched ? Math.max(progressPercent, 8) : 0, steps };
}

export default function Home() {
  const [projectTitle, setProjectTitle] = useState(defaultTitle);
  const [idea, setIdea] = useState(defaultIdea);
  const [targetUsers, setTargetUsers] = useState("College students balancing multiple courses");
  const [techStackPreference, setTechStackPreference] = useState("React + Vite frontend, FastAPI backend, Postgres-ready schema");
  const [requiredFeatures, setRequiredFeatures] = useState(
    "Study goal intake, weekly plan generator, focus session tracker, progress dashboard, API health check",
  );
  const [repoDescription, setRepoDescription] = useState(
    "StudyPilot — an autonomous MVP generated by MVPilot.",
  );
  const [primaryRulesUrl, setPrimaryRulesUrl] = useState(defaultReferenceUrl);
  const [additionalSources, setAdditionalSources] = useState<AdditionalSource[]>([]);
  const [nextSourceType, setNextSourceType] = useState<AdditionalSourceType>("url");
  const [repoPreference, setRepoPreference] = useState<RepoPreference>("create_new_repo");
  const [requestedRepoName, setRequestedRepoName] = useState("mvpilot-generated-studypilot");
  const [existingRepoUrl, setExistingRepoUrl] = useState("");
  const [visibility, setVisibility] = useState<RepoVisibility>("private");
  const [githubConnectionId, setGithubConnectionId] = useState<string | null>(null);
  const [githubUsername, setGithubUsername] = useState<string | null>(null);
  const [githubStatus, setGithubStatus] = useState<GithubStatus>("not_connected");
  const [githubOAuthConfig, setGithubOAuthConfig] = useState<GithubOAuthConfig | null>(null);
  const [githubMessage, setGithubMessage] = useState("Connect GitHub so MVPilot can create the project repo through backend OAuth.");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskDetail, setTaskDetail] = useState<TaskDetail | null>(null);
  const [submitState, setSubmitState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [message, setMessage] = useState("Ready for preflight. Add the build brief and launch MVPilot.");

  useEffect(() => {
    const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL;

    async function restoreGithubSession() {
      const { connectionId, username, error } = readGithubCallbackState();
      const wasConnecting = window.sessionStorage.getItem("mvpilot_github_connecting") === "true";

      if (error) {
        setGithubMessage(error);
        setGithubStatus("error");
        clearGithubConnectingFlags();
        window.history.replaceState({}, "", window.location.pathname);
        return;
      }

      if (connectionId && apiBaseUrl) {
        const status = await fetchGithubStatus(apiBaseUrl, connectionId);
        if (status.connected) {
          setGithubConnectionId(connectionId);
          setGithubUsername(status.username || username);
          setGithubStatus("connected");
          setGithubMessage(
            status.username || username
              ? `Connected as ${status.username || username}.`
              : "GitHub connected. Repo creation is ready.",
          );
          clearGithubConnectingFlags();
          window.history.replaceState({}, "", window.location.pathname);
          return;
        }
      }

      if (wasConnecting) {
        if (githubConnectingExpired()) {
          clearGithubConnectingFlags();
          setGithubStatus("error");
          setGithubMessage("GitHub sign-in timed out. Click Retry GitHub to try again.");
          return;
        }
        setGithubStatus("connecting");
        setGithubMessage("Finishing GitHub sign-in...");
      }
    }

    void restoreGithubSession();

    const storedTaskId = window.sessionStorage.getItem("mvpilot_task_id");
    if (storedTaskId) {
      window.setTimeout(() => {
        setTaskId(storedTaskId);
        setSubmitState("sent");
        setMessage("Resuming flight telemetry for your in-progress run...");
      }, 0);
    }

    if (!apiBaseUrl) return;

    fetch(`${apiBaseUrl}/api/auth/github/config`, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((config) => {
        if (!config) return;
        const typed = config as GithubOAuthConfig;
        setGithubOAuthConfig(typed);
        const { connectionId, error } = readGithubCallbackState();
        if (!connectionId && !error && !typed.oauthConfigured) {
          setGithubStatus("error");
          setGithubMessage(
            `GitHub OAuth is not configured. Add ${(typed.missingEnv || ["GITHUB_OAUTH_CLIENT_ID", "GITHUB_OAUTH_CLIENT_SECRET"]).join(", ")} and register ${typed.redirectUri || "the backend callback URL"}.`,
          );
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (githubStatus !== "connecting") return;
    const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL;
    if (!apiBaseUrl) {
      window.setTimeout(() => {
        setGithubStatus("error");
        setGithubMessage("Missing NEXT_PUBLIC_AGENT_API_URL. Start the FastAPI backend before connecting GitHub.");
        clearGithubConnectingFlags();
      }, 0);
      return;
    }
    const githubApiBaseUrl = apiBaseUrl;

    let cancelled = false;

    async function pollGithubConnection() {
      if (cancelled) return;

      if (githubConnectingExpired()) {
        clearGithubConnectingFlags();
        setGithubStatus("error");
        setGithubMessage("GitHub sign-in timed out. Click Retry GitHub to try again.");
        return;
      }

      const { connectionId, username, error } = readGithubCallbackState();
      if (error) {
        setGithubStatus("error");
        setGithubMessage(error);
        clearGithubConnectingFlags();
        window.history.replaceState({}, "", window.location.pathname);
        return;
      }

      if (!connectionId) return;

      const status = await fetchGithubStatus(githubApiBaseUrl, connectionId);
      if (cancelled) return;
      if (!status.connected) return;

      setGithubConnectionId(connectionId);
      setGithubUsername(status.username || username);
      setGithubStatus("connected");
      setGithubMessage(
        status.username || username
          ? `Connected as ${status.username || username}.`
          : "GitHub connected. Repo creation is ready.",
      );
      clearGithubConnectingFlags();
      window.history.replaceState({}, "", window.location.pathname);
    }

    void pollGithubConnection();
    const intervalId = window.setInterval(() => void pollGithubConnection(), 1500);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [githubStatus]);

  useEffect(() => {
    if (!taskId || !process.env.NEXT_PUBLIC_AGENT_API_URL || submitState === "idle") {
      return;
    }

    let cancelled = false;
    const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL;

    async function pollTask() {
      try {
        const response = await fetch(`${apiBaseUrl}/agent/tasks/${taskId}`, { cache: "no-store" });
        if (!response.ok) return;
        const detail = await response.json() as TaskDetail;
        if (cancelled) return;
        setTaskDetail(detail);
        const status = detail.task.status;
        if (status === "completed") {
          setSubmitState("sent");
          setMessage("MVP landed. Review the evidence, build log, and GitHub links below.");
          return;
        }
        if (status === "failed") {
          setSubmitState("error");
          const failedStep = [...(detail.agent_steps ?? [])].reverse().find((step) => step.status === "failed");
          const rawMessage =
            failedStep?.message
            || detail.final_report?.summary
            || "Workflow failed. Check the failed flight-stage event below.";
          setMessage(formatWorkflowError(rawMessage));
          return;
        }
      } catch {
        if (!cancelled) setMessage("Waiting for task telemetry from the orchestrator...");
      }
      if (!cancelled) window.setTimeout(pollTask, 1500);
    }

    pollTask();
    return () => {
      cancelled = true;
    };
  }, [submitState, taskId]);

  const payloadPreview = useMemo(
    () => ({
      title: projectTitle.trim() || "Untitled MVP idea",
      idea,
      github_connected: Boolean(githubConnectionId),
      github_connection_id: githubConnectionId,
      rulesUrl: primaryRulesUrl.trim() || null,
      referenceUrls: additionalSources
        .filter((source): source is Extract<AdditionalSource, { type: "url" }> => source.type === "url")
        .map((source) => source.value.trim())
        .filter(Boolean),
      additional_files: additionalSources
        .filter((source): source is Extract<AdditionalSource, { type: "file" }> => source.type === "file" && Boolean(source.file))
        .map((source) => ({
          name: source.file?.name,
          type: source.file?.type || "unknown",
          size: source.file?.size,
        })),
      repoPreference,
      repoName: repoPreference === "create_new_repo" ? requestedRepoName.trim() || null : null,
      repoDescription: repoDescription.trim() || null,
      repoUrl: repoPreference === "use_existing_repo" ? existingRepoUrl.trim() || null : null,
      visibility,
      targetUsers: targetUsers.trim() || null,
      techStackPreference: techStackPreference.trim() || null,
      requiredFeatures: requiredFeatures
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
      source: "mvpilot_frontend",
    }),
    [additionalSources, existingRepoUrl, githubConnectionId, idea, primaryRulesUrl, projectTitle, repoDescription, repoPreference, requestedRepoName, requiredFeatures, targetUsers, techStackPreference, visibility],
  );

  const buildTimeline = taskDetail?.build_timeline ?? [];
  const mvpPlan = taskDetail?.mvp_plan ?? taskDetail?.final_report?.mvp_plan ?? null;
  const rawMvpDelivery = taskDetail?.mvp_delivery ?? taskDetail?.final_report?.mvp_delivery ?? null;
  const rawMvpValidation = taskDetail?.mvp_validation ?? taskDetail?.final_report?.mvp_validation ?? null;
  const mvpValidation =
    rawMvpValidation && Array.isArray(rawMvpValidation.checks) && rawMvpValidation.checks.length > 0
      ? rawMvpValidation
      : null;
  const mvpDelivery =
    rawMvpDelivery && typeof rawMvpDelivery.validation_passed === "boolean" ? rawMvpDelivery : null;

  const hasLaunched = Boolean(taskId);
  const runFailed = submitState === "error";
  const stageState = deriveFlightStageState(taskDetail, hasLaunched, runFailed);
  const progressPercent = stageState.progressPercent;
  const activeStopIndex = stageState.activeStopIndex;
  const steps = stageState.steps;
  const repoUrl = taskDetail?.final_report?.links?.repoUrl || taskDetail?.final_report?.repo?.url || null;
  const commitUrl = taskDetail?.final_report?.links?.commitUrl || null;
  const buildLogPath = taskDetail?.final_report?.links?.buildLogPath || "docs/BUILD_LOG.md";
  const architectureDocPath = taskDetail?.final_report?.links?.architectureDocPath || "docs/ARCHITECTURE.md";
  const ragEvidence = taskDetail?.build_context?.evidence ?? [];
  const toolCalls = taskDetail?.tool_calls ?? [];
  const buildLogArtifact = taskDetail?.generated_artifacts?.find((artifact) => artifact.name === "docs/BUILD_LOG.md");
  const additionalUrlCount = payloadPreview.referenceUrls.length;
  const additionalFileCount = payloadPreview.additional_files.length;
  const activityLog = useMemo(() => {
    const rows: WorkflowStep[] = [];
    if (githubConnectionId) {
      rows.push({
        node_name: "github_connected",
        status: "completed",
        message: githubUsername ? `GitHub connected as ${githubUsername}.` : "GitHub connected.",
        flight_stage: "preflight",
        agent: "github",
      });
    }
    rows.push(...(taskDetail?.agent_steps ?? []));
    if (repoUrl) {
      rows.push({
        node_name: "repo_ready",
        status: "completed",
        message: "Final GitHub repository link is ready.",
        flight_stage: "landed",
        agent: "github",
      });
    }
    return rows;
  }, [githubConnectionId, githubUsername, repoUrl, taskDetail?.agent_steps]);

  function connectGitHub() {
    const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL;

    if (!apiBaseUrl) {
      setGithubMessage("Missing NEXT_PUBLIC_AGENT_API_URL. Start the FastAPI backend URL before connecting GitHub.");
      setGithubStatus("error");
      return;
    }

    const params = new URLSearchParams({
      return_to: githubReturnTo(),
    });

    setGithubStatus("connecting");
    setGithubMessage("Opening GitHub OAuth...");
    markGithubConnecting();
    window.location.assign(`${apiBaseUrl}/api/auth/github/login?${params.toString()}`);
  }

  const oauthReady = githubOAuthConfig?.oauthConfigured === true;

  function resetFlight() {
    setTaskId(null);
    setTaskDetail(null);
    setSubmitState("idle");
    setMessage("Ready for preflight. Add the build brief and launch MVPilot.");
    window.sessionStorage.removeItem("mvpilot_task_id");
  }

  async function disconnectGitHub() {
    const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL;
    if (apiBaseUrl && githubConnectionId) {
      await fetch(`${apiBaseUrl}/api/auth/github/disconnect?github_connection_id=${encodeURIComponent(githubConnectionId)}`, {
        method: "POST",
      }).catch(() => undefined);
    }
    setGithubConnectionId(null);
    setGithubUsername(null);
    setGithubStatus("not_connected");
    window.sessionStorage.removeItem("mvpilot_github_connection_id");
    window.sessionStorage.removeItem("mvpilot_github_username");
    clearGithubConnectingFlags();
    setGithubMessage("GitHub disconnected for this browser session.");
  }

  function addSource() {
    setAdditionalSources((sources) => [
      ...sources,
      nextSourceType === "url"
        ? { id: Date.now(), type: "url", value: "" }
        : { id: Date.now(), type: "file", file: null },
    ]);
  }

  function removeSource(id: number) {
    setAdditionalSources((sources) => sources.filter((source) => source.id !== id));
  }

  function updateAdditionalUrl(id: number, value: string) {
    setAdditionalSources((sources) =>
      sources.map((source) => source.id === id && source.type === "url" ? { ...source, value } : source),
    );
  }

  function updateAdditionalFile(id: number, event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setAdditionalSources((sources) =>
      sources.map((source) => source.id === id && source.type === "file" ? { ...source, file } : source),
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!projectTitle.trim()) {
      setSubmitState("error");
      setMessage("Add a project title before starting the run.");
      return;
    }

    if (repoPreference === "create_new_repo" && !requestedRepoName.trim()) {
      setSubmitState("error");
      setMessage("Add a repo name or choose an existing repository before launch.");
      return;
    }

    if (repoPreference === "use_existing_repo" && !existingRepoUrl.trim()) {
      setSubmitState("error");
      setMessage("Add the existing GitHub repo URL before launch.");
      return;
    }

    setSubmitState("sending");
    setMessage("Flight plan filed. Sending the build brief to Person 1's orchestrator...");

    const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL;

    if (!apiBaseUrl) {
      setSubmitState("error");
      setMessage("NEXT_PUBLIC_AGENT_API_URL is missing. Configure the FastAPI backend URL before launching a real MVPilot build.");
      return;
    }

    const githubStatus = await fetchGithubStatus(apiBaseUrl, githubConnectionId);
    if (!githubStatus.connected) {
      setSubmitState("error");
      setGithubStatus("error");
      setGithubMessage("GitHub is not connected. Complete backend OAuth before launching repo actions.");
      setMessage("Connect GitHub before launch so the backend can create or update the repo without frontend tokens.");
      return;
    }
    setGithubStatus("connected");
    setGithubUsername(githubStatus.username || githubUsername);
    setGithubMessage(githubStatus.username ? `Connected as ${githubStatus.username}.` : "GitHub connection ready.");

    const formData = new FormData();
    formData.append("title", payloadPreview.title);
    formData.append("idea", idea);
    if (primaryRulesUrl.trim()) formData.append("rulesUrl", primaryRulesUrl.trim());
    formData.append("repoPreference", repoPreference);
    formData.append("visibility", visibility);
    if (repoDescription.trim()) formData.append("repoDescription", repoDescription.trim());
    if (repoPreference === "create_new_repo") formData.append("repoName", requestedRepoName.trim());
    if (repoPreference === "use_existing_repo") formData.append("repoUrl", existingRepoUrl.trim());
    if (targetUsers.trim()) formData.append("targetUsers", targetUsers.trim());
    if (techStackPreference.trim()) formData.append("techStackPreference", techStackPreference.trim());
    payloadPreview.requiredFeatures.forEach((feature) => formData.append("requiredFeatures", feature));
    formData.append("source", "mvpilot_frontend");

    if (githubConnectionId) {
      formData.append("github_connected", "true");
      formData.append("github_connection_id", githubConnectionId);
    }

    additionalSources.forEach((source) => {
      if (source.type === "url" && source.value.trim()) {
        formData.append("referenceUrls", source.value.trim());
      }

      if (source.type === "file" && source.file) {
        formData.append("additional_files", source.file);
      }
    });

    try {
      const response = await fetch(`${apiBaseUrl}/api/orchestrator/start-project`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = typeof payload.detail === "string" ? payload.detail : `Agent returned ${response.status}`;
        throw new Error(detail);
      }

      const data = await response.json();
      const launchedTaskId = data.task_id ?? data.id ?? "sent-without-task-id";
      setTaskId(launchedTaskId);
      window.sessionStorage.setItem("mvpilot_task_id", launchedTaskId);
      setSubmitState("sent");
      setMessage("MVPilot is airborne. Polling live flight telemetry from the orchestrator.");
    } catch (error) {
      setSubmitState("error");
      setMessage(error instanceof Error ? error.message : "Could not reach the agent endpoint.");
    }
  }

  return (
    <main className="min-h-screen bg-[#0a0e17] text-[#dfe2ef] [background-image:linear-gradient(rgba(0,242,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,242,255,0.03)_1px,transparent_1px)] [background-size:20px_20px]">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-5 py-5 lg:px-8">
        <header className="flex h-auto flex-col gap-4 border-b border-[#3a494b]/50 bg-[#0f131c]/80 pb-5 backdrop-blur-xl lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-md border border-[#00f2ff]/30 bg-[#00f2ff]/10 font-mono text-sm font-black text-[#00f2ff] shadow-[0_0_15px_rgba(0,242,255,0.16)]">MP</div>
              <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#4edea3]">Mission Control</p>
            </div>
            <h1 className="mt-3 max-w-4xl text-4xl font-bold tracking-normal text-[#e1fdff] lg:text-5xl">
              MVPilot Flight Control
            </h1>
            <p className="mt-3 max-w-3xl text-base leading-7 text-[#b9cacb]">
              Connect GitHub, describe an MVP, create a repo, watch Nemotron/OpenClaw build it, then open the finished GitHub project.
            </p>
          </div>
          <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/70 p-4 backdrop-blur-xl">
            <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#849495]">System</p>
            <div className="mt-2 flex items-center gap-3">
              <span className="h-2.5 w-2.5 rounded-full bg-[#4edea3] shadow-[0_0_10px_rgba(78,222,163,0.9)]" />
              <p className="font-mono text-sm font-semibold text-[#4edea3]">FRONTEND STABLE</p>
            </div>
            <p className="mt-2 font-mono text-xs text-[#b9cacb]">Task: {taskId ? taskId : "NOT LAUNCHED"}</p>
          </div>
        </header>

        <section className="overflow-hidden rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur-xl">
          <div className="relative min-h-[560px] p-5 lg:p-7">
            <div className="pointer-events-none absolute -left-48 -top-72 h-[620px] w-[620px] rounded-full bg-[conic-gradient(from_0deg_at_50%_50%,rgba(0,242,255,0.12)_0deg,transparent_90deg)]" />

            {!hasLaunched ? (
              <form onSubmit={handleSubmit} className="relative z-10 grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
                <section className="rounded-lg border border-[#00f2ff]/15 bg-[#0f131c]/80 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Preflight Brief</p>
                      <h2 className="mt-2 text-2xl font-semibold tracking-normal text-[#dfe2ef]">Launch Parameters</h2>
                    </div>
                    <StatusPill status={submitState === "error" ? "Failed" : "Ready"} />
                  </div>

                  <label className="mt-5 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="project-title">Project title</label>
                  <input id="project-title" type="text" required value={projectTitle} onChange={(event) => setProjectTitle(event.target.value)} placeholder="StudyPilot" className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />

                  <label className="mt-4 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="idea">Project idea</label>
                  <textarea id="idea" value={idea} onChange={(event) => setIdea(event.target.value)} rows={4} className="mt-2 w-full resize-none rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />

                  <label className="mt-4 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="target-users">Target users</label>
                  <input id="target-users" type="text" value={targetUsers} onChange={(event) => setTargetUsers(event.target.value)} placeholder="Who is this MVP for?" className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />

                  <label className="mt-4 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="tech-stack">Tech stack preference</label>
                  <input id="tech-stack" type="text" value={techStackPreference} onChange={(event) => setTechStackPreference(event.target.value)} placeholder="React, FastAPI, Postgres..." className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />

                  <label className="mt-4 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="required-features">Required features (comma-separated)</label>
                  <textarea id="required-features" value={requiredFeatures} onChange={(event) => setRequiredFeatures(event.target.value)} rows={2} className="mt-2 w-full resize-none rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />

                  <label className="mt-4 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="repo-description">Repo description</label>
                  <input id="repo-description" type="text" value={repoDescription} onChange={(event) => setRepoDescription(event.target.value)} placeholder="Optional GitHub repository description" className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />

                  <section className="mt-5 rounded-lg border border-[#3a494b]/70 bg-[#1c1f29]/70 p-3">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#4edea3]">GitHub Link</p>
                        <p className="mt-1 text-sm leading-5 text-[#b9cacb]">Backend OAuth creates the session and stores the encrypted GitHub token server-side.</p>
                      </div>
                      {githubConnectionId ? (
                        <button type="button" onClick={disconnectGitHub} className="rounded border border-[#00f2ff]/50 px-3 py-2 font-mono text-xs font-bold uppercase tracking-widest text-[#00f2ff] transition hover:bg-[#00f2ff]/10">Disconnect</button>
                      ) : (
                        <button
                          type="button"
                          onClick={connectGitHub}
                          disabled={githubStatus === "connecting" || (githubOAuthConfig !== null && !oauthReady)}
                          title={oauthReady ? "Sign in with GitHub OAuth" : "OAuth is not configured in backend .env"}
                          className="rounded bg-[#00f2ff] px-3 py-2 font-mono text-xs font-bold uppercase tracking-widest text-[#00363a] transition hover:shadow-[0_0_15px_rgba(0,242,255,0.28)] disabled:cursor-not-allowed disabled:bg-[#31353f] disabled:text-[#849495]"
                        >
                          {githubStatus === "connecting"
                            ? "Connecting..."
                            : githubStatus === "error"
                              ? "Retry GitHub"
                              : "Connect GitHub"}
                        </button>
                      )}
                    </div>
                    {githubOAuthConfig && !oauthReady ? (
                      <p className="mt-3 rounded border border-[#ffb4ab]/40 bg-[#93000a]/20 px-3 py-2 text-xs leading-5 text-[#ffb4ab]">
                        OAuth is not set up on the backend. Add{" "}
                        {(githubOAuthConfig.missingEnv || ["GITHUB_OAUTH_CLIENT_ID", "GITHUB_OAUTH_CLIENT_SECRET"]).join(", ")} to .env and register{" "}
                        <span className="font-mono">{githubOAuthConfig.redirectUri}</span> in a GitHub OAuth app.
                      </p>
                    ) : null}
                    <div className={`mt-3 flex items-start justify-between gap-3 rounded border bg-[#0a0e17] px-3 py-2 ${githubMessage.toLowerCase().includes("not configured") ? "border-[#ffb4ab]/40" : "border-[#3a494b]"}`}>
                      <div>
                        <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#849495]">{githubStatusLabel(githubStatus)}</p>
                        <p className={`mt-1 text-sm leading-6 ${githubStatus === "error" ? "text-[#ffb4ab]" : "text-[#b9cacb]"}`}>{githubMessage}</p>
                        {githubUsername && <p className="mt-1 font-mono text-xs text-[#4edea3]">@{githubUsername}</p>}
                      </div>
                      <StatusPill status={githubStatusPill(githubStatus)} />
                    </div>
                  </section>

                  <section className="mt-5 rounded-lg border border-[#3a494b]/70 bg-[#1c1f29]/70 p-3">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#4edea3]">Repository</p>
                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      <label className="block">
                        <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#b9cacb]">Action</span>
                        <select value={repoPreference} onChange={(event) => setRepoPreference(event.target.value as RepoPreference)} className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-xs font-bold uppercase tracking-widest text-[#dfe2ef] outline-none focus:border-[#00f2ff]">
                          <option value="create_new_repo">Create new repo</option>
                          <option value="use_existing_repo">Use existing repo</option>
                        </select>
                      </label>
                      <label className="block">
                        <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#b9cacb]">Visibility</span>
                        <select value={visibility} onChange={(event) => setVisibility(event.target.value as RepoVisibility)} className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-xs font-bold uppercase tracking-widest text-[#dfe2ef] outline-none focus:border-[#00f2ff]">
                          <option value="private">Private</option>
                          <option value="public">Public</option>
                        </select>
                      </label>
                    </div>
                    {repoPreference === "create_new_repo" ? (
                      <label className="mt-3 block">
                        <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#b9cacb]">Repo name</span>
                        <input type="text" value={requestedRepoName} onChange={(event) => setRequestedRepoName(event.target.value)} placeholder="mvpilot-generated-your-idea" className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff]" />
                        <p className="mt-1 font-mono text-[10px] text-[#849495]">Must start with mvpilot-generated- so repo actions stay scoped to MVPilot output.</p>
                      </label>
                    ) : (
                      <label className="mt-3 block">
                        <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#b9cacb]">Existing repo URL</span>
                        <input type="url" value={existingRepoUrl} onChange={(event) => setExistingRepoUrl(event.target.value)} placeholder="https://github.com/user/repo" className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff]" />
                      </label>
                    )}
                  </section>

                  <label className="mt-4 block font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]" htmlFor="primary-rules-url">Optional reference URL</label>
                  <input id="primary-rules-url" type="url" value={primaryRulesUrl} onChange={(event) => setPrimaryRulesUrl(event.target.value)} placeholder="https://example.com/product-docs (optional)" className="mt-2 w-full rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff] focus:ring-1 focus:ring-[#00f2ff]/40" />
                  <p className="mt-1 font-mono text-[10px] text-[#849495]">Optional product or rules context. Delete to skip external RAG fetch.</p>

                  <div className="mt-5 rounded-lg border border-[#3a494b]/70 bg-[#1c1f29]/70 p-3">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#b9cacb]">Additional sources</p>
                        <p className="mt-1 text-sm leading-5 text-[#849495]">Add extra URLs or files only when the agent needs more context.</p>
                      </div>
                      <div className="flex gap-2">
                        <select value={nextSourceType} onChange={(event) => setNextSourceType(event.target.value as AdditionalSourceType)} className="rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-xs font-bold uppercase tracking-widest text-[#dfe2ef] outline-none focus:border-[#00f2ff]">
                          <option value="url">URL</option>
                          <option value="file">File</option>
                        </select>
                        <button type="button" onClick={addSource} className="rounded bg-[#00f2ff] px-3 py-2 font-mono text-xs font-black text-[#00363a] transition hover:shadow-[0_0_15px_rgba(0,242,255,0.28)]" aria-label="Add additional source">+</button>
                      </div>
                    </div>

                    {additionalSources.length > 0 && (
                      <div className="mt-3 grid gap-3">
                        {additionalSources.map((source) => (
                          <div key={source.id} className="rounded border border-[#3a494b] bg-[#0a0e17] p-3">
                            <div className="flex items-center justify-between gap-3">
                              <p className="font-mono text-xs font-bold uppercase tracking-widest text-[#00f2ff]">Additional {source.type}</p>
                              <button type="button" onClick={() => removeSource(source.id)} className="rounded border border-[#3a494b] px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-[#b9cacb] transition hover:border-[#00f2ff]/50">Remove</button>
                            </div>
                            {source.type === "url" ? (
                              <input type="url" value={source.value} onChange={(event) => updateAdditionalUrl(source.id, event.target.value)} placeholder="https://example.com/extra-docs" className="mt-2 w-full rounded border border-[#3a494b] bg-[#181b25] px-3 py-2 font-mono text-sm leading-6 text-[#e1fdff] outline-none transition focus:border-[#00f2ff]" />
                            ) : (
                              <input type="file" accept=".pdf,.md,.txt,.doc,.docx,.json,.csv" onChange={(event) => updateAdditionalFile(source.id, event)} className="mt-2 w-full rounded border border-dashed border-[#3a494b] bg-[#181b25] px-3 py-3 text-sm text-[#b9cacb] file:mr-4 file:rounded file:border-0 file:bg-[#00f2ff] file:px-3 file:py-2 file:font-mono file:text-xs file:font-bold file:uppercase file:tracking-widest file:text-[#00363a]" />
                            )}
                            {source.type === "file" && source.file && (
                              <p className="mt-2 font-mono text-xs text-[#849495]">{source.file.name} ({Math.ceil(source.file.size / 1024)} KB)</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <button type="submit" className="mt-5 w-full rounded bg-[#00f2ff] px-4 py-3 font-mono text-xs font-black uppercase tracking-widest text-[#00363a] transition hover:shadow-[0_0_18px_rgba(0,242,255,0.35)] disabled:cursor-not-allowed disabled:bg-[#31353f] disabled:text-[#849495]">
                    Launch MVPilot
                  </button>

                  <p className="mt-3 rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 font-mono text-xs leading-5 text-[#b9cacb]">{message}</p>
                </section>

                <section className="relative min-h-[500px] overflow-hidden rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/50 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                  <div className="absolute inset-0 opacity-70 [background-image:linear-gradient(rgba(0,242,255,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(0,242,255,0.04)_1px,transparent_1px)] [background-size:20px_20px]" />
                  <div className="relative z-10">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Flight Path</p>
                    <h2 className="mt-2 text-2xl font-semibold tracking-normal text-[#e1fdff]">Parameters staged on runway</h2>
                    <p className="mt-2 max-w-xl text-sm leading-6 text-[#b9cacb]">After launch, this panel becomes the live checkpoint bar. The plane advances as the Person 1 orchestrator reports progress.</p>
                  </div>
                  <div className="absolute left-8 right-8 top-1/2 h-px bg-[#3a494b]" />
                  <div className="absolute left-8 right-8 top-1/2 flex -translate-y-1/2 justify-between">
                    {flightStops.map((stop) => (
                      <div key={stop.phase} className="flex flex-col items-center gap-3">
                        <div className="h-4 w-4 rounded-full border-4 border-[#0f131c] bg-[#31353f]" />
                        <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#849495]">{stop.phase}</p>
                      </div>
                    ))}
                  </div>
                  <PlaneMarker progress={2} />
                </section>
              </form>
            ) : (
              <section className="relative z-10 grid gap-5">
                <div className="rounded-lg border border-[#00f2ff]/15 bg-[#0f131c]/85 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">
                        {runFailed ? "Flight aborted" : "Autopilot Active"}
                      </p>
                      <h2 className="mt-2 text-2xl font-semibold tracking-normal text-[#e1fdff]">{payloadPreview.title}</h2>
                      <p className={`mt-2 max-w-3xl font-mono text-xs leading-5 ${runFailed ? "text-[#ffb4ab]" : "text-[#b9cacb]"}`}>{message}</p>
                    </div>
                    <div className="flex flex-col items-start gap-3 lg:items-end">
                      <div className="text-left lg:text-right">
                        <p className="font-mono text-4xl font-bold text-[#00f2ff]">{progressPercent}%</p>
                        <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#849495]">Completion</p>
                      </div>
                      {runFailed ? (
                        <button
                          type="button"
                          onClick={resetFlight}
                          className="rounded border border-[#00f2ff]/50 px-3 py-2 font-mono text-xs font-bold uppercase tracking-widest text-[#00f2ff] transition hover:bg-[#00f2ff]/10"
                        >
                          New launch
                        </button>
                      ) : null}
                    </div>
                  </div>

                  <div className="mt-12 overflow-x-auto px-2 pb-5 pt-12">
                    <div className="relative min-w-[900px]">
                      <div className="absolute left-0 right-0 top-8 h-0.5 bg-[#3a494b]" />
                      <div className="absolute left-0 top-8 h-0.5 bg-[#4edea3] shadow-[0_0_10px_rgba(78,222,163,0.65)] transition-all duration-700" style={{ width: `calc(${progressPercent}% - 12px)` }} />
                      <PlaneMarker progress={progressPercent} />
                      <div className="grid grid-cols-6 gap-4">
                        {steps.map((step) => (
                          <div key={step.phase} className="relative flex flex-col items-center pt-3 text-center">
                            <div className={`z-10 h-5 w-5 rounded-full border-4 border-[#0f131c] ${step.status === "Complete" ? "bg-[#4edea3] shadow-[0_0_12px_rgba(78,222,163,0.65)]" : step.status === "Running" ? "bg-[#00f2ff] shadow-[0_0_20px_rgba(0,242,255,0.7)]" : step.status === "Failed" ? "bg-[#ffb4ab] shadow-[0_0_16px_rgba(255,180,171,0.45)]" : "bg-[#31353f]"}`} />
                            <p className={`mt-4 font-mono text-[11px] font-bold uppercase tracking-widest ${step.status === "Running" ? "text-[#00f2ff]" : step.status === "Complete" ? "text-[#4edea3]" : step.status === "Failed" ? "text-[#ffb4ab]" : "text-[#849495]"}`}>{step.phase}</p>
                            <p className="mt-1 text-xs leading-5 text-[#b9cacb]">{step.title}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid gap-5 lg:grid-cols-2">
                  <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">OpenClaw Build Timeline</p>
                    <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{taskDetail?.runtime === "openclaw" ? "OpenClaw orchestration" : "LangGraph orchestration"}</h3>
                    <div className="mt-4 grid max-h-80 gap-2 overflow-y-auto">
                      {buildTimeline.map((phase) => (
                        <div key={phase.id} className="rounded border border-[#3a494b] bg-[#0a0e17] p-3">
                          <div className="flex items-center justify-between gap-3">
                            <p className="font-mono text-xs font-bold uppercase tracking-widest text-[#e1fdff]">{phase.title}</p>
                            <StatusPill status={phase.status === "completed" ? "Complete" : phase.status === "running" ? "Running" : phase.status === "failed" ? "Failed" : "Pending"} />
                          </div>
                          {phase.detail ? <p className="mt-2 text-sm leading-6 text-[#b9cacb]">{phase.detail}</p> : null}
                        </div>
                      ))}
                      {!buildTimeline.length ? <p className="text-sm text-[#849495]">Timeline phases appear as the builder progresses.</p> : null}
                    </div>
                  </div>
                  <div className="rounded-lg border border-[#4edea3]/20 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#4edea3]">MVP Plan</p>
                    <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{mvpPlan?.title || payloadPreview.title}</h3>
                    {mvpPlan?.features?.length ? (
                      <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-[#b9cacb]">{mvpPlan.features.map((f) => <li key={f}>{f}</li>)}</ul>
                    ) : <p className="mt-3 text-sm text-[#849495]">Plan details appear after scope + repo planning.</p>}
                  </div>
                </div>

                {mvpValidation ? (
                  <div className="grid gap-5 lg:grid-cols-2">
                    <div className="rounded-lg border border-[#ffb86c]/25 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                      <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#ffb86c]">MVP Delivery</p>
                      <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{mvpDelivery?.project_title || mvpPlan?.title || payloadPreview.title}</h3>
                      {mvpDelivery?.model_modes?.length ? (
                        <p className="mt-2 font-mono text-xs text-[#849495]">Model modes: {mvpDelivery.model_modes.join(", ")}</p>
                      ) : null}
                      {mvpDelivery?.completed_features?.length ? (
                        <div className="mt-3">
                          <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#4edea3]">Completed</p>
                          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-[#b9cacb]">{mvpDelivery.completed_features.map((f) => <li key={f}>{f}</li>)}</ul>
                        </div>
                      ) : null}
                      {mvpDelivery?.mocked_features?.length ? (
                        <div className="mt-3">
                          <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#00f2ff]">Mocked</p>
                          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-[#b9cacb]">{mvpDelivery.mocked_features.map((f) => <li key={f}>{f}</li>)}</ul>
                        </div>
                      ) : null}
                      {mvpDelivery?.pending_features?.length ? (
                        <div className="mt-3">
                          <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#ffb4ab]">Pending</p>
                          <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-[#b9cacb]">{mvpDelivery.pending_features.map((f) => <li key={f}>{f}</li>)}</ul>
                        </div>
                      ) : null}
                    </div>
                    <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                      <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Idea Validation</p>
                      <div className="mt-2 flex items-center gap-3">
                        <h3 className="text-xl font-semibold text-[#dfe2ef]">{mvpValidation?.passed ? "Aligned with your idea" : "Review recommended"}</h3>
                        <StatusPill status={mvpValidation?.passed ? "Complete" : "Failed"} />
                      </div>
                      <div className="mt-4 grid max-h-56 gap-2 overflow-y-auto">
                        {(mvpValidation?.checks ?? []).map((check) => (
                          <div key={check.name} className="rounded border border-[#3a494b] bg-[#0a0e17] p-3">
                            <p className="font-mono text-xs font-bold uppercase tracking-widest text-[#e1fdff]">{check.name.replaceAll("_", " ")}</p>
                            <p className="mt-1 text-sm text-[#b9cacb]">{check.detail}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : null}

                <div className="grid gap-5 lg:grid-cols-3">
                  <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] lg:col-span-3">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Agent Activity Log</p>
                        <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{activityLog.length ? `${activityLog.length} events recorded` : "Waiting for first event"}</h3>
                      </div>
                      <p className="font-mono text-xs uppercase tracking-widest text-[#849495]">Live build timeline</p>
                    </div>
                    <div className="mt-4 grid max-h-72 gap-2 overflow-y-auto">
                      {activityLog.map((event, index) => (
                        <div key={`${event.node_name}-${index}`} className="grid gap-3 rounded border border-[#3a494b] bg-[#0a0e17] p-3 md:grid-cols-[170px_1fr_auto] md:items-center">
                          <div>
                            <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#4edea3]">{event.agent || "agent"}</p>
                            <p className="mt-1 font-mono text-xs text-[#849495]">{event.node_name}</p>
                          </div>
                          <p className="text-sm leading-6 text-[#b9cacb]">{event.message}</p>
                          <StatusPill status={event.status === "failed" ? "Failed" : event.status === "completed" || event.status === "success" ? "Complete" : "Running"} />
                        </div>
                      ))}
                      {!activityLog.length && <p className="rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 text-xs text-[#849495]">No agent events yet.</p>}
                    </div>
                  </div>

                  <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Radar Evidence</p>
                    <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{ragEvidence.length ? `${ragEvidence.length} chunks retrieved` : "Waiting for context"}</h3>
                    <div className="mt-4 grid max-h-56 gap-2 overflow-y-auto text-xs text-[#b9cacb]">
                      {ragEvidence.slice(0, 4).map((evidence, index) => (
                        <div key={`${evidence.source}-${index}`} className="rounded border border-[#3a494b] bg-[#0a0e17] p-3">
                          <p className="font-mono font-bold uppercase tracking-widest text-[#4edea3]">{evidence.docType || evidence.doc_type || "source"} · {typeof evidence.score === "number" ? evidence.score.toFixed(2) : "n/a"}</p>
                          <p className="mt-1 font-mono text-[#849495]">{evidence.source}</p>
                          <p className="mt-2 line-clamp-3 leading-5">{evidence.content || evidence.text}</p>
                        </div>
                      ))}
                      {!ragEvidence.length && (
                        <div className="rounded border border-[#3a494b] bg-[#0a0e17] p-3 font-mono">
                          <p>Primary URL: {primaryRulesUrl}</p>
                          <p>Extra URLs: {additionalUrlCount}</p>
                          <p>Extra files: {additionalFileCount}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Tool Calls</p>
                    <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{steps[activeStopIndex].title}</h3>
                    <p className="mt-3 text-sm leading-6 text-[#b9cacb]">{steps[activeStopIndex].detail}</p>
                    <div className="mt-4 grid max-h-48 gap-2 overflow-y-auto">
                      {toolCalls.slice(-5).map((call, index) => (
                        <div key={`${call.tool}-${index}`} className="rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2">
                          <div className="flex items-center justify-between gap-3">
                            <p className="font-mono text-xs font-bold text-[#dfe2ef]">{call.tool || "tool"}</p>
                            <StatusPill status={call.status === "success" ? "Complete" : call.status === "failed" ? "Failed" : "Running"} />
                          </div>
                          {call.summary && <p className="mt-2 text-xs leading-5 text-[#b9cacb]">{call.summary}</p>}
                        </div>
                      ))}
                      {!toolCalls.length && <p className="rounded border border-[#3a494b] bg-[#0a0e17] px-3 py-2 text-xs text-[#849495]">No tool calls reported yet.</p>}
                    </div>
                  </div>

                  <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                    <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Landing Zone</p>
                    <h3 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{repoUrl ? "GitHub repo ready" : "GitHub repo pending"}</h3>
                    {repoUrl ? (
                      <div className="mt-4 flex flex-wrap gap-2">
                        <a href={repoUrl} target="_blank" rel="noreferrer" className="inline-flex rounded bg-[#00f2ff] px-3 py-2 font-mono text-xs font-black uppercase tracking-widest text-[#00363a]">Open repo</a>
                        {commitUrl && <a href={commitUrl} target="_blank" rel="noreferrer" className="inline-flex rounded border border-[#00f2ff]/50 px-3 py-2 font-mono text-xs font-black uppercase tracking-widest text-[#00f2ff]">Commit</a>}
                      </div>
                    ) : (
                      <p className="mt-3 text-sm leading-6 text-[#b9cacb]">When Person 1 returns repo_url, this card links to the generated project, README, demo script, and pitch.</p>
                    )}
                    <div className="mt-4 grid gap-2 font-mono text-xs text-[#b9cacb]">
                      <p>Build log: {buildLogPath}</p>
                      <p>Architecture: {architectureDocPath}</p>
                      {buildLogArtifact?.summary && <p>{buildLogArtifact.summary}</p>}
                    </div>
                    {taskId && <p className="mt-4 rounded border border-[#00f2ff]/20 bg-[#00f2ff]/10 px-3 py-2 font-mono text-xs font-bold text-[#00f2ff]">Task ID: {taskId}</p>}
                  </div>
                </div>
              </section>
            )}
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-3">
          <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/70 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur-xl">
            <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Run Summary</p>
            <h2 className="mt-2 text-xl font-semibold text-[#dfe2ef]">{payloadPreview.title}</h2>
            <p className="mt-3 text-sm leading-6 text-[#b9cacb]">{idea}</p>
          </div>

          <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/70 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur-xl">
            <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">RAG Inputs</p>
            <h2 className="mt-2 text-xl font-semibold text-[#dfe2ef]">Allowed source material</h2>
            <div className="mt-3 grid gap-2">
              {sourceTypes.map((source) => (
                <div key={source} className="rounded border border-[#3a494b] bg-[#0f131c] px-3 py-2 text-sm text-[#b9cacb]">{source}</div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-[#00f2ff]/15 bg-[#181b25]/70 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur-xl">
            <p className="font-mono text-[11px] font-bold uppercase tracking-widest text-[#00f2ff]">Person 1 Handoff</p>
            <h2 className="mt-2 text-xl font-semibold text-[#dfe2ef]">Orchestrator contract</h2>
            <div className="mt-3 space-y-2 text-sm leading-6 text-[#b9cacb]">
            <p>POST /api/orchestrator/start-project receives idea, optional references, repoPreference, repoName, repoDescription, repoUrl, and visibility.</p>
              <p>Backend returns task_id, then GET /agent/tasks/:id streams RAG evidence, tool calls, build logs, and final GitHub links.</p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
