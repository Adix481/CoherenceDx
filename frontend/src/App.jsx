import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

import {
  Cpu,
  Waves,
  Brain,
  UploadCloud,
  SlidersHorizontal,
  Activity,
  ChevronRight,
  Loader2,
  WifiOff,
  Wifi,
  RefreshCcw,
} from "lucide-react";

import {
  getHealth,
  getEvaluation,
  getBenchmarks,
  predict,
} from "./api/api";

import {
  FEATURE_GROUPS,
  FEATURE_ORDER,
  HEART_FEATURES,
  LABEL_MAP,
  HEART_LABEL_MAP,
  SAMPLE_INPUT,
  SAMPLE_HEART_INPUT,
} from "./features";

/* ============================================================
   MODEL OPTIONS
   ============================================================ */

const MODEL_OPTIONS = [
  {
    id: "svm",
    label: "SVM",
  },
  {
    id: "quantum_vqc",
    label: "Quantum VQC (4 qubits)",
  },
  {
    id: "hybrid_vqc",
    label: "Hybrid VQC (4 qubits)",
  },
];

/* ============================================================
   COLORS
   ============================================================ */

const C = {
  bg: "#070B14",
  panel: "#0E1524",
  panel2: "#141F35",
  line: "#22304A",
  lineSoft: "#1A2740",
  teal: "#52DCC9",
  tealDim: "#2E7A6E",
  amber: "#F0A857",
  amberDim: "#8A6335",
  violet: "#9B8CF2",
  hi: "#EAF1FB",
  mid: "#8CA0C4",
  low: "#4E5E7D",
  good: "#6FD3A0",
  risk: "#E8735D",
};

/* ============================================================
   FONTS
   ============================================================ */

const fontStyles = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,900&family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.font-display {
  font-family: 'Fraunces', serif;
  font-optical-sizing: auto;
}

.font-body {
  font-family: 'Space Grotesk', sans-serif;
}

.font-mono {
  font-family: 'IBM Plex Mono', monospace;
}
`;

/* ============================================================
   PIPELINE
   ============================================================ */

const pipelineStages = [
  {
    n: "01",
    title: "Data ingestion",
    body: "Load the patient record and required clinical or diagnostic features.",
    icon: UploadCloud,
  },
  {
    n: "02",
    title: "Preprocessing",
    body: "Normalize the input features using the trained preprocessing pipeline.",
    icon: SlidersHorizontal,
  },
  {
    n: "03",
    title: "Quantum encoding",
    body: "Reduced features are converted into quantum rotation angles.",
    icon: Waves,
  },
  {
    n: "04",
    title: "Variational circuit",
    body: "A 4-qubit variational quantum circuit processes the encoded features.",
    icon: Cpu,
  },
  {
    n: "05",
    title: "Inference",
    body: "The Flask backend executes the selected trained model.",
    icon: Activity,
  },
  {
    n: "06",
    title: "Prediction",
    body: "The system returns a disease prediction and probability score.",
    icon: Brain,
  },
];

/* ============================================================
   DELIVERABLES
   ============================================================ */

const deliverables = [
  [
    "01",
    "Data ingestion & preprocessing",
    "Dataset-specific preprocessing and feature transformation",
  ],
  [
    "02",
    "Feature reduction",
    "Classical preprocessing followed by quantum feature encoding",
  ],
  [
    "03",
    "Quantum VQC",
    "4-qubit variational quantum classifier",
  ],
  [
    "04",
    "Hybrid quantum-classical model",
    "Classical preprocessing combined with a quantum circuit",
  ],
  [
    "05",
    "Inference engine",
    "Flask API for live disease prediction",
  ],
  [
    "06",
    "Explainability module",
    "Feature attribution and quantum circuit visualization",
  ],
  [
    "07",
    "Classical baseline",
    "SVM baseline for comparison",
  ],
  [
    "08",
    "Performance evaluation",
    "Accuracy, sensitivity, specificity, precision, F1 and ROC-AUC",
  ],
  [
    "09",
    "Interactive frontend",
    "React/Vite dashboard connected to Flask",
  ],
  [
    "10",
    "Research platform",
    "Unified interface for classical, quantum and hybrid models",
  ],
];

/* ============================================================
   DECORATIVE STRAND
   ============================================================ */

function Strand({ height = 64, id }) {
  const uid = id || "s";

  return (
    <svg
      viewBox="0 0 800 64"
      width="100%"
      height={height}
      preserveAspectRatio="none"
      style={{ display: "block" }}
    >
      <defs>
        <linearGradient id={`fade-${uid}`} x1="0" x2="1">
          <stop offset="0%" stopColor={C.bg} />
          <stop offset="10%" stopColor={C.bg} stopOpacity="0" />
          <stop offset="90%" stopColor={C.bg} stopOpacity="0" />
          <stop offset="100%" stopColor={C.bg} />
        </linearGradient>
      </defs>

      <path
        d="M0,32 C100,4 150,60 250,32 C350,4 400,60 500,32 C600,4 650,60 750,32 L800,32"
        fill="none"
        stroke={C.amber}
        strokeWidth="1.5"
        strokeDasharray="1 7"
        strokeLinecap="round"
        opacity="0.85"
      />

      <path
        d="M0,32 C100,60 150,4 250,32 C350,60 400,4 500,32 C600,60 650,4 750,32 L800,32"
        fill="none"
        stroke={C.teal}
        strokeWidth="1.6"
        opacity="0.9"
      />

      {[110, 260, 410, 560].map((x, i) => (
        <circle
          key={i}
          cx={x}
          cy="32"
          r="3.2"
          fill={C.hi}
          opacity="0.9"
        />
      ))}

      <rect
        x="0"
        y="0"
        width="800"
        height="64"
        fill={`url(#fade-${uid})`}
      />
    </svg>
  );
}

/* ============================================================
   SMALL COMPONENTS
   ============================================================ */

function Eyebrow({ children, color = C.teal }) {
  return (
    <div
      className="font-mono"
      style={{
        fontSize: 11,
        letterSpacing: "0.18em",
        color,
        textTransform: "uppercase",
        marginBottom: 10,
      }}
    >
      {children}
    </div>
  );
}

function Panel({ children, style }) {
  return (
    <div
      style={{
        background: C.panel,
        border: `1px solid ${C.line}`,
        borderRadius: 14,
        padding: "22px 24px",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function Pill({ children, tone = "teal" }) {
  const map = {
    teal: {
      bg: "rgba(82,220,201,0.1)",
      fg: C.teal,
      bd: "rgba(82,220,201,0.35)",
    },
    amber: {
      bg: "rgba(240,168,87,0.1)",
      fg: C.amber,
      bd: "rgba(240,168,87,0.35)",
    },
    good: {
      bg: "rgba(111,211,160,0.1)",
      fg: C.good,
      bd: "rgba(111,211,160,0.35)",
    },
    risk: {
      bg: "rgba(232,115,93,0.1)",
      fg: C.risk,
      bd: "rgba(232,115,93,0.35)",
    },
  }[tone];

  return (
    <span
      className="font-mono"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontSize: 11,
        padding: "4px 10px",
        borderRadius: 999,
        background: map.bg,
        color: map.fg,
        border: `1px solid ${map.bd}`,
        letterSpacing: "0.04em",
      }}
    >
      {children}
    </span>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) {
    return null;
  }

  return (
    <div
      className="font-mono"
      style={{
        background: C.panel2,
        border: `1px solid ${C.line}`,
        borderRadius: 8,
        padding: "10px 12px",
        fontSize: 12,
        color: C.hi,
      }}
    >
      <div
        style={{
          color: C.mid,
          marginBottom: 6,
          whiteSpace: "pre-line",
        }}
      >
        {label}
      </div>

      {payload.map((p, i) => (
        <div
          key={i}
          style={{
            color: p.color,
            display: "flex",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <span>{p.name}</span>

          <span>
            {p.value}
            {typeof p.value === "number" &&
            p.value <= 100 &&
            p.value > 3
              ? "%"
              : ""}
          </span>
        </div>
      ))}
    </div>
  );
};

function Section({
  eyebrow,
  title,
  desc,
  color,
  children,
}) {
  return (
    <div
      style={{
        maxWidth: 1180,
        margin: "0 auto",
        padding: "56px 28px 72px",
      }}
    >
      <div style={{ marginBottom: 28 }}>
        <Eyebrow color={color}>{eyebrow}</Eyebrow>

        <h2
          className="font-display"
          style={{
            color: C.hi,
            fontSize: 28,
            fontWeight: 600,
            margin: "0 0 10px",
          }}
        >
          {title}
        </h2>

        <p
          className="font-body"
          style={{
            color: C.mid,
            fontSize: 14.5,
            maxWidth: 700,
            margin: 0,
            lineHeight: 1.6,
          }}
        >
          {desc}
        </p>
      </div>

      {children}
    </div>
  );
}

function PanelHeader({ label, sub }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div
        className="font-body"
        style={{
          color: C.hi,
          fontSize: 14.5,
          fontWeight: 600,
        }}
      >
        {label}
      </div>

      <div
        className="font-mono"
        style={{
          color: C.low,
          fontSize: 11,
          marginTop: 2,
        }}
      >
        {sub}
      </div>
    </div>
  );
}

function StatTile({ label, value, tone, sub }) {
  return (
    <Panel style={{ textAlign: "left" }}>
      <div
        className="font-mono"
        style={{
          fontSize: 11,
          color: C.mid,
          letterSpacing: "0.05em",
        }}
      >
        {label}
      </div>

      <div
        className="font-display"
        style={{
          fontSize: 30,
          color: tone,
          fontWeight: 600,
          marginTop: 6,
        }}
      >
        {value}
      </div>

      {sub && (
        <div
          className="font-body"
          style={{
            fontSize: 11.5,
            color: C.low,
            marginTop: 2,
          }}
        >
          {sub}
        </div>
      )}
    </Panel>
  );
}

function StatusNotice({
  status,
  error,
  loadingLabel,
}) {
  if (status === "loading") {
    return (
      <Panel
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <Loader2
          size={16}
          color={C.mid}
          className="spin"
        />

        <span
          className="font-body"
          style={{
            color: C.mid,
            fontSize: 13.5,
          }}
        >
          {loadingLabel || "Loading..."}
        </span>
      </Panel>
    );
  }

  if (status === "error") {
    return (
      <Panel
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          borderColor: "rgba(232,115,93,0.4)",
        }}
      >
        <WifiOff size={16} color={C.risk} />

        <span
          className="font-body"
          style={{
            color: C.risk,
            fontSize: 13.5,
          }}
        >
          {error === "OFFLINE"
            ? "Backend Offline"
            : error}
        </span>
      </Panel>
    );
  }

  return null;
}

/* ============================================================
   API HOOK
   ============================================================ */

function useApiResource(fetcher, deps = []) {
  const [state, setState] = useState({
    status: "loading",
    data: null,
    error: null,
  });

  const reload = () => {
    setState({
      status: "loading",
      data: null,
      error: null,
    });

    fetcher()
      .then((data) =>
        setState({
          status: "success",
          data,
          error: null,
        })
      )
      .catch((err) =>
        setState({
          status: "error",
          data: null,
          error:
            err.message === "OFFLINE"
              ? "OFFLINE"
              : err.message,
        })
      );
  };

  useEffect(() => {
    let cancelled = false;

    setState({
      status: "loading",
      data: null,
      error: null,
    });

    fetcher()
      .then((data) => {
        if (!cancelled) {
          setState({
            status: "success",
            data,
            error: null,
          });
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setState({
            status: "error",
            data: null,
            error:
              err.message === "OFFLINE"
                ? "OFFLINE"
                : err.message,
          });
        }
      });

    return () => {
      cancelled = true;
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return {
    ...state,
    reload,
  };
}

/* ============================================================
   NAVIGATION
   ============================================================ */

const TABS = [
  {
    id: "overview",
    label: "Overview",
  },
  {
    id: "prediction",
    label: "Prediction",
  },
  {
    id: "explain",
    label: "Explainability",
  },
  {
    id: "benchmarks",
    label: "Benchmarks",
  },
  {
    id: "evaluation",
    label: "Evaluation",
  },
  {
    id: "deliver",
    label: "Deliverables",
  },
];

function BackendStatus() {
  const { status } = useApiResource(
    getHealth,
    []
  );

  const online = status === "success";

  const label =
    status === "loading"
      ? "Checking backend..."
      : online
      ? "Backend online"
      : "Backend offline";

  const color =
    status === "loading"
      ? C.mid
      : online
      ? C.good
      : C.risk;

  return (
    <span
      title={label}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
      }}
    >
      {online ? (
        <Wifi size={13} color={color} />
      ) : (
        <WifiOff size={13} color={color} />
      )}

      <span
        className="font-mono"
        style={{
          fontSize: 10.5,
          color,
        }}
      >
        {label}
      </span>
    </span>
  );
}

function Nav({ active, setActive }) {
  return (
    <div
      style={{
        position: "sticky",
        top: 0,
        zIndex: 40,
        background: "rgba(7,11,20,0.86)",
        backdropFilter: "blur(10px)",
        borderBottom: `1px solid ${C.line}`,
      }}
    >
      <div
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          padding: "14px 28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 14,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <div
              style={{
                width: 30,
                height: 30,
                borderRadius: 8,
                border: `1px solid ${C.line}`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  width: 14,
                  height: 14,
                  borderRadius: "50%",
                  border: `1.4px solid ${C.teal}`,
                }}
              />

              <div
                style={{
                  position: "absolute",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: C.amber,
                }}
              />
            </div>

            <span
              className="font-display"
              style={{
                color: C.hi,
                fontSize: 19,
                fontWeight: 600,
              }}
            >
              Coherence
              <span style={{ color: C.teal }}>
                Dx
              </span>
            </span>
          </div>

          <BackendStatus />
        </div>

        <div
          className="font-body"
          style={{
            display: "flex",
            gap: 4,
            flexWrap: "wrap",
          }}
        >
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setActive(t.id)}
              style={{
                background:
                  active === t.id
                    ? C.panel2
                    : "transparent",
                color:
                  active === t.id
                    ? C.hi
                    : C.mid,
                border: `1px solid ${
                  active === t.id
                    ? C.line
                    : "transparent"
                }`,
                borderRadius: 8,
                padding: "7px 13px",
                fontSize: 13.5,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   HERO
   ============================================================ */

function HeroGlyph() {
  return (
    <svg
      viewBox="0 0 380 320"
      width="100%"
      height="auto"
      style={{
        maxWidth: 420,
        justifySelf: "center",
      }}
    >
      <defs>
        <radialGradient
          id="glow"
          cx="50%"
          cy="45%"
          r="60%"
        >
          <stop
            offset="0%"
            stopColor="rgba(82,220,201,0.16)"
          />
          <stop
            offset="100%"
            stopColor="rgba(82,220,201,0)"
          />
        </radialGradient>
      </defs>

      <circle
        cx="190"
        cy="140"
        r="150"
        fill="url(#glow)"
      />

      <ellipse
        cx="190"
        cy="140"
        rx="120"
        ry="46"
        fill="none"
        stroke={C.line}
        strokeWidth="1"
      />

      <ellipse
        cx="190"
        cy="140"
        rx="120"
        ry="46"
        fill="none"
        stroke={C.teal}
        strokeWidth="1.3"
        opacity="0.55"
        transform="rotate(60 190 140)"
      />

      <ellipse
        cx="190"
        cy="140"
        rx="120"
        ry="46"
        fill="none"
        stroke={C.amber}
        strokeWidth="1.3"
        opacity="0.5"
        transform="rotate(120 190 140)"
      />

      <circle
        cx="190"
        cy="140"
        r="9"
        fill={C.teal}
      />

      <circle
        cx="310"
        cy="140"
        r="5"
        fill={C.hi}
        opacity="0.85"
      />

      <circle
        cx="70"
        cy="140"
        r="5"
        fill={C.amber}
        opacity="0.85"
      />

      <circle
        cx="190"
        cy="94"
        r="4"
        fill={C.violet}
        opacity="0.85"
      />

      <circle
        cx="190"
        cy="186"
        r="4"
        fill={C.violet}
        opacity="0.85"
      />

      <path
        d="M20,262 L110,262 L128,220 L148,296 L166,240 L182,262 L360,262"
        fill="none"
        stroke={C.mid}
        strokeWidth="1.6"
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity="0.8"
      />

      <text
        x="190"
        y="300"
        textAnchor="middle"
        className="font-mono"
        fontSize="10"
        fill={C.low}
        letterSpacing="0.14em"
      >
        |ψ⟩ QUANTUM DIAGNOSTICS
      </text>
    </svg>
  );
}

/* ============================================================
   OVERVIEW
   ============================================================ */

function Overview({ setActive }) {
  return (
    <div>
      <div
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          padding: "64px 28px 10px",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.25fr 0.9fr",
            gap: 48,
            alignItems: "center",
          }}
          className="hero-grid"
        >
          <div>
            <Eyebrow>
              Hybrid Quantum · Classical Diagnostics
            </Eyebrow>

            <h1
              className="font-display"
              style={{
                fontSize:
                  "clamp(34px, 4.4vw, 54px)",
                lineHeight: 1.06,
                color: C.hi,
                fontWeight: 600,
                margin: 0,
                maxWidth: 620,
              }}
            >
              Two ways of computing,{" "}
              <span
                style={{
                  color: C.teal,
                  fontStyle: "italic",
                }}
              >
                one
              </span>{" "}
              earlier diagnosis.
            </h1>

            <p
              className="font-body"
              style={{
                color: C.mid,
                fontSize: 16.5,
                lineHeight: 1.65,
                marginTop: 20,
                maxWidth: 560,
              }}
            >
              A quantum-enhanced disease detection
              platform combining classical machine
              learning, variational quantum circuits
              and hybrid quantum-classical inference.
            </p>

            <div
              style={{
                display: "flex",
                gap: 12,
                marginTop: 30,
                flexWrap: "wrap",
              }}
            >
              <button
                onClick={() =>
                  setActive("prediction")
                }
                className="font-body"
                style={{
                  background: C.teal,
                  color: "#04211D",
                  border: "none",
                  borderRadius: 9,
                  padding: "12px 20px",
                  fontSize: 14.5,
                  fontWeight: 600,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                Run a prediction
                <ChevronRight size={16} />
              </button>

              <button
                onClick={() =>
                  setActive("benchmarks")
                }
                className="font-body"
                style={{
                  background: "transparent",
                  color: C.hi,
                  border: `1px solid ${C.line}`,
                  borderRadius: 9,
                  padding: "12px 20px",
                  fontSize: 14.5,
                  cursor: "pointer",
                }}
              >
                See benchmarks
              </button>
            </div>

            <div
              style={{
                display: "flex",
                gap: 10,
                marginTop: 26,
                flexWrap: "wrap",
              }}
            >
              <Pill tone="teal">
                4-qubit VQC
              </Pill>

              <Pill tone="amber">
                Heart Disease
              </Pill>

              <Pill tone="amber">
                Breast Cancer
              </Pill>

              <Pill tone="good">
                Classical + Quantum
              </Pill>
            </div>
          </div>

          <HeroGlyph />
        </div>
      </div>

      <Strand id="hero" />

      <div
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          padding: "10px 28px 64px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            marginBottom: 28,
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div>
            <Eyebrow color={C.amber}>
              The pipeline
            </Eyebrow>

            <h2
              className="font-display"
              style={{
                color: C.hi,
                fontSize: 28,
                fontWeight: 600,
                margin: 0,
              }}
            >
              From patient record to prediction
            </h2>
          </div>

          <p
            className="font-body"
            style={{
              color: C.mid,
              fontSize: 14,
              maxWidth: 400,
              margin: 0,
            }}
          >
            The platform supports classical,
            quantum and hybrid inference through
            one unified prediction interface.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(3, 1fr)",
            gap: 16,
          }}
          className="pipeline-grid"
        >
          {pipelineStages.map((s, i) => {
            const Icon = s.icon;
            const quantum =
              i === 2 || i === 3;

            return (
              <Panel key={s.n}>
                <div
                  style={{
                    display: "flex",
                    justifyContent:
                      "space-between",
                    alignItems:
                      "flex-start",
                  }}
                >
                  <span
                    className="font-mono"
                    style={{
                      color: C.low,
                      fontSize: 12,
                    }}
                  >
                    {s.n}
                  </span>

                  <Icon
                    size={17}
                    color={
                      quantum
                        ? C.teal
                        : C.amber
                    }
                  />
                </div>

                <h3
                  className="font-body"
                  style={{
                    color: C.hi,
                    fontSize: 15.5,
                    fontWeight: 600,
                    margin:
                      "14px 0 6px",
                  }}
                >
                  {s.title}
                </h3>

                <p
                  className="font-body"
                  style={{
                    color: C.mid,
                    fontSize: 13.3,
                    lineHeight: 1.55,
                    margin: 0,
                  }}
                >
                  {s.body}
                </p>
              </Panel>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   FEATURE COMPONENTS
   ============================================================ */

function FeatureField({
  name,
  value,
  onChange,
}) {
  return (
    <div style={{ marginBottom: 8 }}>
      <label
        className="font-mono"
        style={{
          fontSize: 10,
          color: C.low,
          display: "block",
          marginBottom: 3,
          textTransform: "uppercase",
        }}
      >
        {name.replace(/_/g, " ")}
      </label>

      <input
        type="number"
        step="any"
        value={value}
        onChange={(e) =>
          onChange(name, e.target.value)
        }
        className="font-mono"
        style={{
          width: "100%",
          background: C.panel2,
          border: `1px solid ${C.line}`,
          borderRadius: 6,
          padding: "7px 9px",
          color: C.hi,
          fontSize: 12.5,
        }}
      />
    </div>
  );
}

function FeatureGroupPanel({
  title,
  keys,
  values,
  onChange,
}) {
  return (
    <Panel>
      <PanelHeader
        label={title}
        sub={`${keys.length} features`}
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(2, 1fr)",
          gap: "0 12px",
        }}
      >
        {keys.map((k) => (
          <FeatureField
            key={k}
            name={k}
            value={values[k]}
            onChange={onChange}
          />
        ))}
      </div>
    </Panel>
  );
}

/* ============================================================
   HEART DISEASE FORM
   ============================================================ */

function HeartDiseaseForm({
  values,
  onChange,
}) {
  const fields = [
    {
      name: "age",
      label: "Age",
      type: "number",
    },
    {
      name: "trestbps",
      label: "Resting Blood Pressure",
      type: "number",
    },
    {
      name: "chol",
      label: "Cholesterol",
      type: "number",
    },
    {
      name: "thalch",
      label: "Maximum Heart Rate",
      type: "number",
    },
    {
      name: "oldpeak",
      label: "ST Depression",
      type: "number",
      step: "any",
    },
    {
      name: "ca",
      label: "Number of Major Vessels",
      type: "number",
      step: "any",
    },
  ];

  return (
    <Panel>
      <PanelHeader
        label="Heart Disease"
        sub="13 clinical features"
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(2, 1fr)",
          gap: 14,
        }}
        className="feature-grid"
      >
        {fields.map((field) => (
          <div key={field.name}>
            <label
              className="font-mono"
              style={{
                fontSize: 10,
                color: C.low,
                display: "block",
                marginBottom: 5,
                textTransform: "uppercase",
              }}
            >
              {field.label}
            </label>

            <input
              type={field.type}
              step={field.step || "1"}
              value={values[field.name]}
              onChange={(e) =>
                onChange(
                  field.name,
                  e.target.value
                )
              }
              className="font-mono"
              style={{
                width: "100%",
                background: C.panel2,
                border: `1px solid ${C.line}`,
                borderRadius: 6,
                padding: "9px 10px",
                color: C.hi,
                fontSize: 12.5,
              }}
            />
          </div>
        ))}

        <HeartSelect
          label="Sex"
          value={values.sex}
          onChange={(value) =>
            onChange("sex", value)
          }
          options={[
            ["Male", "Male"],
            ["Female", "Female"],
          ]}
        />

        <HeartSelect
          label="Chest Pain Type"
          value={values.cp}
          onChange={(value) =>
            onChange("cp", value)
          }
          options={[
            [
              "typical angina",
              "Typical angina",
            ],
            [
              "atypical angina",
              "Atypical angina",
            ],
            [
              "non-anginal",
              "Non-anginal",
            ],
            [
              "asymptomatic",
              "Asymptomatic",
            ],
          ]}
        />

        <HeartSelect
          label="Fasting Blood Sugar"
          value={values.fbs}
          onChange={(value) =>
            onChange("fbs", value)
          }
          options={[
            ["0", "≤ 120 mg/dl"],
            ["1", "> 120 mg/dl"],
          ]}
        />

        <HeartSelect
          label="Resting ECG"
          value={values.restecg}
          onChange={(value) =>
            onChange("restecg", value)
          }
          options={[
            ["normal", "Normal"],
            [
              "lv hypertrophy",
              "LV hypertrophy",
            ],
            [
              "st-t abnormality",
              "ST-T abnormality",
            ],
          ]}
        />

        <HeartSelect
          label="Exercise Induced Angina"
          value={values.exang}
          onChange={(value) =>
            onChange("exang", value)
          }
          options={[
            ["0", "No"],
            ["1", "Yes"],
          ]}
        />

        <HeartSelect
          label="Slope"
          value={values.slope}
          onChange={(value) =>
            onChange("slope", value)
          }
          options={[
            [
              "upsloping",
              "Upsloping",
            ],
            ["flat", "Flat"],
            [
              "downsloping",
              "Downsloping",
            ],
          ]}
        />

        <HeartSelect
          label="Thalassemia"
          value={values.thal}
          onChange={(value) =>
            onChange("thal", value)
          }
          options={[
            ["normal", "Normal"],
            [
              "fixed defect",
              "Fixed defect",
            ],
            [
              "reversable defect",
              "Reversible defect",
            ],
          ]}
        />
      </div>
    </Panel>
  );
}

function HeartSelect({
  label,
  value,
  onChange,
  options,
}) {
  return (
    <div>
      <label
        className="font-mono"
        style={{
          fontSize: 10,
          color: C.low,
          display: "block",
          marginBottom: 5,
          textTransform: "uppercase",
        }}
      >
        {label}
      </label>

      <select
        value={value}
        onChange={(e) =>
          onChange(e.target.value)
        }
        className="font-body"
        style={{
          width: "100%",
          background: C.panel2,
          border: `1px solid ${C.line}`,
          borderRadius: 6,
          padding: "9px 10px",
          color: C.hi,
          fontSize: 12.5,
        }}
      >
        <option value="">
          Select...
        </option>

        {options.map(([value, label]) => (
          <option
            key={value}
            value={value}
          >
            {label}
          </option>
        ))}
      </select>
    </div>
  );
}

/* ============================================================
   PREDICTION FLOW
   ============================================================ */

function PredictionFlow({
  disease,
  model,
}) {
  const isBreast =
    disease === "breast_cancer";

  const steps = isBreast
    ? [
        {
          label: "30 features",
          sub: "WDBC record",
          tone: C.amber,
        },
        {
          label: "StandardScaler",
          sub: "normalize",
          tone: C.mid,
        },
        {
          label: "PCA",
          sub: "30 → 4",
          tone: C.mid,
        },
        {
          label: "[0, π]",
          sub: "angle encoding",
          tone: C.teal,
        },
        {
          label:
            model === "svm"
              ? "SVM"
              : "4-qubit VQC",
          sub:
            model === "svm"
              ? "classical"
              : model === "hybrid_vqc"
              ? "hybrid"
              : "quantum",
          tone:
            model === "svm"
              ? C.good
              : model === "hybrid_vqc"
              ? C.violet
              : C.teal,
        },
        {
          label: "Prediction",
          sub: "Benign / Malignant",
          tone: C.good,
        },
      ]
    : [
        {
          label: "13 features",
          sub: "clinical record",
          tone: C.amber,
        },
        {
          label: "Encoding",
          sub: "preprocess",
          tone: C.mid,
        },
        {
          label: "Feature map",
          sub: "transform",
          tone: C.mid,
        },
        {
          label: "4-qubit VQC",
          sub:
            model === "hybrid_vqc"
              ? "hybrid"
              : "quantum",
          tone:
            model === "hybrid_vqc"
              ? C.violet
              : C.teal,
        },
        {
          label:
            model === "svm"
              ? "SVM"
              : "Classifier",
          sub: "inference",
          tone:
            model === "svm"
              ? C.good
              : C.violet,
        },
        {
          label: "Prediction",
          sub: "Disease / No Disease",
          tone: C.good,
        },
      ];

  return (
    <Panel
      style={{
        padding: "16px 20px",
        marginBottom: 20,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 4,
        }}
      >
        {steps.map((s, i) => (
          <React.Fragment key={s.label}>
            <div
              style={{
                display: "flex",
                flexDirection:
                  "column",
                alignItems: "center",
                minWidth: 92,
                padding: "4px 6px",
              }}
            >
              <span
                className="font-mono"
                style={{
                  fontSize: 11.5,
                  color: s.tone,
                  fontWeight: 600,
                  textAlign: "center",
                }}
              >
                {s.label}
              </span>

              <span
                className="font-body"
                style={{
                  fontSize: 10.5,
                  color: C.low,
                  marginTop: 2,
                  textAlign: "center",
                }}
              >
                {s.sub}
              </span>
            </div>

            {i <
              steps.length - 1 && (
              <ChevronRight
                size={14}
                color={C.low}
                style={{
                  flexShrink: 0,
                }}
              />
            )}
          </React.Fragment>
        ))}
      </div>
    </Panel>
  );
}

/* ============================================================
   GAUGE
   ============================================================ */

function Gauge3({
  value,
  color,
}) {
  const safeValue = Math.max(
    0,
    Math.min(1, Number(value) || 0)
  );

  const r = 54;
  const c = 2 * Math.PI * r;
  const pct = safeValue * c;

  return (
    <svg
      width="140"
      height="140"
      viewBox="0 0 140 140"
    >
      <circle
        cx="70"
        cy="70"
        r={r}
        fill="none"
        stroke={C.lineSoft}
        strokeWidth="12"
      />

      <circle
        cx="70"
        cy="70"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="12"
        strokeDasharray={`${pct} ${c}`}
        strokeLinecap="round"
        transform="rotate(-90 70 70)"
      />

      <text
        x="70"
        y="66"
        textAnchor="middle"
        className="font-display"
        fontSize="26"
        fontWeight="600"
        fill={C.hi}
      >
        {Math.round(
          safeValue * 100
        )}
        %
      </text>

      <text
        x="70"
        y="86"
        textAnchor="middle"
        className="font-mono"
        fontSize="9.5"
        fill={C.mid}
        letterSpacing="0.08em"
      >
        PROBABILITY
      </text>
    </svg>
  );
}

/* ============================================================
   EMPTY STATES
   ============================================================ */

function emptyFeatureState() {
  return FEATURE_ORDER.reduce(
    (acc, key) => {
      acc[key] = "";
      return acc;
    },
    {}
  );
}

function emptyHeartState() {
  return {
    age: "",
    sex: "",
    cp: "",
    trestbps: "",
    chol: "",
    fbs: "",
    restecg: "",
    thalch: "",
    exang: "",
    oldpeak: "",
    slope: "",
    ca: "",
    thal: "",
  };
}

/* ============================================================
   PREDICTION
   ============================================================ */

function Prediction() {
  const [disease, setDisease] =
    useState("breast_cancer");

  const [values, setValues] =
    useState(emptyFeatureState());

  const [heartValues, setHeartValues] =
    useState(emptyHeartState());

  const [model, setModel] =
    useState(MODEL_OPTIONS[0].id);

  const [state, setState] =
    useState({
      status: "idle",
      data: null,
      error: null,
    });

  function handleChange(
    name,
    value
  ) {
    setValues((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  function handleHeartChange(
    name,
    value
  ) {
    setHeartValues((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  function handleDiseaseChange(
    value
  ) {
    setDisease(value);

    setState({
      status: "idle",
      data: null,
      error: null,
    });

    setModel(
      MODEL_OPTIONS[0].id
    );
  }

  function fillSample() {
    if (
      disease === "breast_cancer"
    ) {
      setValues({
        ...SAMPLE_INPUT,
      });
    } else {
      setHeartValues({
        ...SAMPLE_HEART_INPUT,
      });
    }

    setState({
      status: "idle",
      data: null,
      error: null,
    });
  }

  function clearForm() {
    if (
      disease === "breast_cancer"
    ) {
      setValues(
        emptyFeatureState()
      );
    } else {
      setHeartValues(
        emptyHeartState()
      );
    }

    setState({
      status: "idle",
      data: null,
      error: null,
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();

    setState({
      status: "loading",
      data: null,
      error: null,
    });

    try {
      let data;

      /* ------------------------------------------------------
         BREAST CANCER
         ------------------------------------------------------ */

      if (
        disease === "breast_cancer"
      ) {
        const features = {};

        for (
          const key of FEATURE_ORDER
        ) {
          const raw =
            values[key];

          if (
            raw === "" ||
            raw === null ||
            raw === undefined ||
            Number.isNaN(
              Number(raw)
            )
          ) {
            setState({
              status: "error",
              data: null,
              error:
                "All 30 breast cancer feature values are required.",
            });

            return;
          }

          features[key] =
            Number(raw);
        }

        data = await predict({
          disease:
            "breast_cancer",
          model,
          features,
        });
      }

      /* ------------------------------------------------------
         HEART DISEASE
         ------------------------------------------------------ */

      else {
        const features = {};

        for (
          const key of HEART_FEATURES
        ) {
          const raw =
            heartValues[key];

          if (
            raw === "" ||
            raw === null ||
            raw === undefined
          ) {
            setState({
              status: "error",
              data: null,
              error:
                "All 13 heart disease feature values are required.",
            });

            return;
          }

          features[key] =
            raw;
        }

        data = await predict({
          disease:
            "heart_disease",
          model,
          features,
        });
      }

      setState({
        status: "success",
        data,
        error: null,
      });
    } catch (err) {
      setState({
        status: "error",
        data: null,
        error:
          err.message === "OFFLINE"
            ? "OFFLINE"
            : err.message,
      });
    }
  }

  const result = state.data;

  const rawLabel =
    result?.label ??
    result?.prediction;

  let label;

  if (
    disease === "breast_cancer"
  ) {
    label =
      typeof rawLabel ===
      "number"
        ? LABEL_MAP[
            rawLabel
          ]
        : LABEL_MAP[
            rawLabel
          ] ?? rawLabel;
  } else {
    label =
      typeof rawLabel ===
      "number"
        ? HEART_LABEL_MAP[
            rawLabel
          ]
        : HEART_LABEL_MAP[
            rawLabel
          ] ?? rawLabel;
  }

  const probability =
    result?.probability ??
    result?.confidence ??
    null;

  const modelName =
    result?.model ??
    result?.model_name ??
    MODEL_OPTIONS.find(
      (m) => m.id === model
    )?.label;

  const isRisk =
    disease ===
    "breast_cancer"
      ? label === "Malignant"
      : label === "Disease";

  return (
    <Section
      eyebrow="Inference"
      title="Score a patient"
      color={C.amber}
      desc={
        disease ===
        "breast_cancer"
          ? "Enter the 30 WDBC features and select a prediction model."
          : "Enter the clinical features and select a prediction model."
      }
    >
      {/* DISEASE SELECTOR */}

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          marginBottom: 20,
          flexWrap: "wrap",
        }}
      >
        <span
          className="font-mono"
          style={{
            color: C.mid,
            fontSize: 12,
          }}
        >
          DISEASE
        </span>

        <select
          value={disease}
          onChange={(e) =>
            handleDiseaseChange(
              e.target.value
            )
          }
          className="font-body"
          style={{
            background: C.panel2,
            color: C.hi,
            border: `1px solid ${C.line}`,
            borderRadius: 9,
            padding: "11px 14px",
            fontSize: 13.5,
          }}
        >
          <option value="breast_cancer">
            Breast Cancer
          </option>

          <option value="heart_disease">
            Heart Disease
          </option>
        </select>
      </div>

      <PredictionFlow
        disease={disease}
        model={model}
      />

      {/* INPUT FORM */}

      {disease ===
      "breast_cancer" ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(3, 1fr)",
            gap: 16,
          }}
          className="feature-grid"
        >
          <FeatureGroupPanel
            title="Mean"
            keys={
              FEATURE_GROUPS.mean
            }
            values={values}
            onChange={
              handleChange
            }
          />

          <FeatureGroupPanel
            title="Standard Error"
            keys={
              FEATURE_GROUPS.se
            }
            values={values}
            onChange={
              handleChange
            }
          />

          <FeatureGroupPanel
            title="Worst"
            keys={
              FEATURE_GROUPS.worst
            }
            values={values}
            onChange={
              handleChange
            }
          />
        </div>
      ) : (
        <HeartDiseaseForm
          values={heartValues}
          onChange={
            handleHeartChange
          }
        />
      )}

      {/* CONTROLS */}

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          flexWrap: "wrap",
          marginTop: 20,
        }}
      >
        <div>
          <div
            className="font-mono"
            style={{
              color: C.low,
              fontSize: 10,
              marginBottom: 5,
              textTransform:
                "uppercase",
            }}
          >
            Prediction Model
          </div>

          <select
            value={model}
            onChange={(e) =>
              setModel(
                e.target.value
              )
            }
            className="font-body"
            style={{
              background: C.panel2,
              color: C.hi,
              border: `1px solid ${C.line}`,
              borderRadius: 9,
              padding: "11px 14px",
              fontSize: 13.5,
              minWidth: 230,
              cursor: "pointer",
            }}
          >
            {MODEL_OPTIONS.map(
              (m) => (
                <option
                  key={m.id}
                  value={m.id}
                >
                  {m.label}
                </option>
              )
            )}
          </select>
        </div>

        <button
          type="button"
          onClick={handleSubmit}
          disabled={
            state.status ===
            "loading"
          }
          className="font-body"
          style={{
            background: C.teal,
            color: "#04211D",
            border: "none",
            borderRadius: 9,
            padding: "12px 20px",
            fontSize: 14.5,
            fontWeight: 600,
            cursor:
              state.status ===
              "loading"
                ? "not-allowed"
                : "pointer",
            opacity:
              state.status ===
              "loading"
                ? 0.7
                : 1,
            marginTop: 19,
          }}
        >
          {state.status ===
          "loading"
            ? "Predicting..."
            : "Predict"}
        </button>

        <button
          type="button"
          onClick={fillSample}
          className="font-body"
          style={{
            background:
              "transparent",
            color: C.hi,
            border: `1px solid ${C.line}`,
            borderRadius: 9,
            padding: "12px 16px",
            fontSize: 13.5,
            cursor: "pointer",
            marginTop: 19,
          }}
        >
          Fill example input
        </button>

        <button
          type="button"
          onClick={clearForm}
          className="font-body"
          style={{
            background:
              "transparent",
            color: C.mid,
            border: `1px solid ${C.line}`,
            borderRadius: 9,
            padding: "12px 16px",
            fontSize: 13.5,
            cursor: "pointer",
            marginTop: 19,
          }}
        >
          Clear
        </button>
      </div>

      {/* RESULT */}

      <div
        style={{
          marginTop: 20,
        }}
      >
        {state.status ===
          "idle" && (
          <Panel>
            <span
              className="font-body"
              style={{
                color: C.mid,
                fontSize: 13.5,
              }}
            >
              Fill in the required
              features and submit
              to see a prediction.
            </span>
          </Panel>
        )}

        {state.status ===
          "loading" && (
          <StatusNotice
            status="loading"
            loadingLabel="Sending record to the backend..."
          />
        )}

        {state.status ===
          "error" && (
          <StatusNotice
            status="error"
            error={state.error}
          />
        )}

        {state.status ===
          "success" && (
          <Panel>
            <PanelHeader
              label="Prediction result"
              sub={`Model: ${modelName}`}
            />

            <div
              style={{
                display: "flex",
                gap: 28,
                alignItems: "center",
                flexWrap: "wrap",
              }}
            >
              {probability !==
                null &&
                probability !==
                  undefined && (
                  <Gauge3
                    value={Number(
                      probability
                    )}
                    color={
                      isRisk
                        ? C.risk
                        : C.good
                    }
                  />
                )}

              <div>
                <div
                  className="font-display"
                  style={{
                    fontSize: 30,
                    fontWeight: 600,
                    color: isRisk
                      ? C.risk
                      : C.good,
                  }}
                >
                  {label ??
                    "Unknown"}
                </div>

                {probability !==
                  null &&
                  probability !==
                    undefined && (
                    <div
                      className="font-mono"
                      style={{
                        fontSize: 12.5,
                        color: C.mid,
                        marginTop: 6,
                      }}
                    >
                      Probability:{" "}
                      {(
                        Number(
                          probability
                        ) * 100
                      ).toFixed(
                        1
                      )}
                      %
                    </div>
                  )}

                <div
                  style={{
                    marginTop: 12,
                  }}
                >
                  <Pill
                    tone={
                      isRisk
                        ? "risk"
                        : "good"
                    }
                  >
                    {modelName}
                  </Pill>
                </div>
              </div>
            </div>
          </Panel>
        )}
      </div>
    </Section>
  );
}

/* ============================================================
   EXPLAINABILITY
   ============================================================ */

const featureImportance = [
  {
    feature: "PC1",
    value: 0.187,
  },
  {
    feature: "PC2",
    value: 0.163,
  },
  {
    feature: "PC3",
    value: 0.121,
  },
  {
    feature: "PC4",
    value: 0.098,
  },
];

function Explain() {
  return (
    <Section
      eyebrow="Explainability"
      title="Why the model decided what it decided"
      color={C.violet}
      desc="Inspect the reduced quantum representation and the structure of the variational circuit."
    >
      <Panel
        style={{
          marginBottom: 20,
        }}
      >
        <div
          style={{
            display: "flex",
            gap: 12,
            alignItems:
              "flex-start",
          }}
        >
          <Brain
            size={18}
            color={C.violet}
            style={{
              marginTop: 2,
              flexShrink: 0,
            }}
          />

          <p
            className="font-body"
            style={{
              color: C.mid,
              fontSize: 13.3,
              lineHeight: 1.65,
              margin: 0,
            }}
          >
            The hybrid pipeline first
            transforms the input using
            classical preprocessing before
            passing the reduced representation
            to the quantum circuit.
          </p>
        </div>
      </Panel>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "1.1fr 0.9fr",
          gap: 20,
        }}
        className="responsive-2col"
      >
        <Panel>
          <div
            style={{
              display: "flex",
              justifyContent:
                "space-between",
              alignItems:
                "flex-start",
              flexWrap: "wrap",
              gap: 8,
            }}
          >
            <PanelHeader
              label="Feature attribution"
              sub="Principal-component contribution"
            />

            <Pill tone="amber">
              Illustrative
            </Pill>
          </div>

          <ResponsiveContainer
            width="100%"
            height={260}
          >
            <BarChart
              data={
                featureImportance
              }
              layout="vertical"
              margin={{
                left: 10,
                right: 20,
              }}
            >
              <CartesianGrid
                stroke={C.lineSoft}
                horizontal={false}
              />

              <XAxis
                type="number"
                tick={{
                  fill: C.low,
                  fontSize: 11,
                }}
                axisLine={{
                  stroke: C.line,
                }}
                tickLine={false}
              />

              <YAxis
                type="category"
                dataKey="feature"
                width={60}
                tick={{
                  fill: C.mid,
                  fontSize: 12,
                }}
                axisLine={{
                  stroke: C.line,
                }}
                tickLine={false}
              />

              <Tooltip
                content={
                  <CustomTooltip />
                }
              />

              <Bar
                dataKey="value"
                name="Contribution"
                radius={[
                  0,
                  5,
                  5,
                  0,
                ]}
              >
                {featureImportance.map(
                  (_, i) => (
                    <Cell
                      key={i}
                      fill={
                        i === 0
                          ? C.violet
                          : C.tealDim
                      }
                    />
                  )
                )}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <p
            className="font-body"
            style={{
              color: C.low,
              fontSize: 11.8,
              marginTop: 8,
              lineHeight: 1.5,
            }}
          >
            The chart represents the
            reduced feature space used by
            the quantum model.
          </p>
        </Panel>

        <Panel>
          <PanelHeader
            label="Quantum Circuit"
            sub="4-qubit hybrid quantum-classical circuit"
          />

          <div
            style={{
              marginTop: 16,
              borderRadius: 10,
              overflow: "hidden",
              border: `1px solid ${C.line}`,
              background: "#ffffff",
            }}
          >
            <img
              src="/hybrid_quantum_circuit.jpeg"
              alt="Hybrid Quantum-Classical Circuit"
              style={{
                display: "block",
                width: "100%",
                height: "auto",
              }}
            />
          </div>

          <p
            className="font-body"
            style={{
              color: C.mid,
              fontSize: 12.8,
              marginTop: 10,
              lineHeight: 1.6,
            }}
          >
            Four qubits process the encoded
            feature representation before the
            final measurement is converted into
            a classification probability.
          </p>
        </Panel>
      </div>
    </Section>
  );
}

/* ============================================================
   BENCHMARKS
   ============================================================ */

function Benchmarks() {
  const {
    status,
    data,
    error,
    reload,
  } = useApiResource(
    getBenchmarks,
    []
  );

  const models =
    data?.models ?? [];

  return (
    <Section
      eyebrow="Benchmarking"
      title="Quantum vs. Classical Performance"
      color={C.good}
      desc="Compare the classical, quantum and hybrid models using the benchmark results returned by Flask."
    >
      {status !==
        "success" && (
        <StatusNotice
          status={status}
          error={error}
          loadingLabel="Loading benchmarks..."
        />
      )}

      {status ===
        "success" &&
        models.length === 0 && (
          <Panel>
            <span
              className="font-body"
              style={{
                color: C.mid,
                fontSize: 13.5,
              }}
            >
              No benchmark data
              returned yet.
            </span>
          </Panel>
        )}

      {status ===
        "success" &&
        models.length > 0 && (
          <>
            <Panel>
              <PanelHeader
                label="Accuracy comparison"
                sub="Performance across returned benchmark models"
              />

              <ResponsiveContainer
                width="100%"
                height={360}
              >
                <BarChart
                  data={models}
                  margin={{
                    top: 20,
                    right: 20,
                    left: 0,
                    bottom: 50,
                  }}
                >
                  <CartesianGrid
                    stroke={
                      C.lineSoft
                    }
                    vertical={false}
                  />

                  <XAxis
                    dataKey="name"
                    tick={{
                      fill: C.low,
                      fontSize: 10,
                    }}
                    axisLine={{
                      stroke: C.line,
                    }}
                    tickLine={false}
                    angle={-15}
                    textAnchor="end"
                    interval={0}
                  />

                  <YAxis
                    domain={[
                      0,
                      100,
                    ]}
                    tick={{
                      fill: C.low,
                      fontSize: 11,
                    }}
                    axisLine={{
                      stroke: C.line,
                    }}
                    tickLine={false}
                    tickFormatter={(
                      value
                    ) =>
                      `${value}%`
                    }
                  />

                  <Tooltip
                    content={
                      <CustomTooltip />
                    }
                  />

                  <Bar
                    dataKey="accuracy"
                    name="Accuracy"
                    radius={[
                      5,
                      5,
                      0,
                      0,
                    ]}
                  >
                    {models.map(
                      (
                        model,
                        index
                      ) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={
                            model.type ===
                            "quantum"
                              ? C.violet
                              : model.type ===
                                "hybrid"
                              ? C.amber
                              : C.teal
                          }
                        />
                      )
                    )}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <div
                className="font-body"
                style={{
                  display: "flex",
                  justifyContent:
                    "center",
                  gap: 24,
                  marginTop: 8,
                  fontSize: 12,
                  color: C.mid,
                  flexWrap: "wrap",
                }}
              >
                <span>
                  <span
                    style={{
                      display:
                        "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background:
                        C.teal,
                      marginRight: 7,
                    }}
                  />
                  Classical
                </span>

                <span>
                  <span
                    style={{
                      display:
                        "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background:
                        C.violet,
                      marginRight: 7,
                    }}
                  />
                  Quantum
                </span>

                <span>
                  <span
                    style={{
                      display:
                        "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background:
                        C.amber,
                      marginRight: 7,
                    }}
                  />
                  Hybrid
                </span>
              </div>
            </Panel>

            <Panel
              style={{
                marginTop: 20,
                padding: 0,
                overflow:
                  "hidden",
              }}
            >
              <div
                style={{
                  overflowX:
                    "auto",
                }}
              >
                <table
                  className="font-body"
                  style={{
                    width: "100%",
                    borderCollapse:
                      "collapse",
                    minWidth: 850,
                  }}
                >
                  <thead>
                    <tr
                      style={{
                        borderBottom:
                          `1px solid ${C.line}`,
                        textAlign:
                          "left",
                      }}
                    >
                      {[
                        "Disease",
                        "Model",
                        "Type",
                        "Accuracy",
                        "Sensitivity",
                        "Specificity",
                        "Precision",
                        "F1",
                        "ROC-AUC",
                      ].map(
                        (h) => (
                          <th
                            key={h}
                            style={th}
                          >
                            {h}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>

                  <tbody>
                    {models.map(
                      (
                        m,
                        i
                      ) => (
                        <tr
                          key={`${m.name}-${i}`}
                          style={{
                            borderBottom:
                              i <
                              models.length -
                                1
                                ? `1px solid ${C.lineSoft}`
                                : "none",
                          }}
                        >
                          <td
                            style={{
                              ...td,
                              color:
                                C.hi,
                            }}
                          >
                            {m.disease ??
                              "—"}
                          </td>

                          <td
                            style={{
                              ...td,
                              fontWeight: 600,
                              color:
                                m.type ===
                                "quantum"
                                  ? C.violet
                                  : m.type ===
                                    "hybrid"
                                  ? C.amber
                                  : C.hi,
                            }}
                          >
                            {m.name ??
                              "—"}
                          </td>

                          <td
                            style={
                              td
                            }
                          >
                            <span
                              style={{
                                display:
                                  "inline-block",
                                padding:
                                  "4px 9px",
                                borderRadius:
                                  999,
                                fontSize: 11,
                                fontWeight: 600,
                                color:
                                  m.type ===
                                  "quantum"
                                    ? C.violet
                                    : m.type ===
                                      "hybrid"
                                    ? C.amber
                                    : C.teal,
                                border:
                                  `1px solid ${
                                    m.type ===
                                    "quantum"
                                      ? C.violet
                                      : m.type ===
                                        "hybrid"
                                      ? C.amber
                                      : C.teal
                                  }`,
                              }}
                            >
                              {m.type ===
                              "quantum"
                                ? "QUANTUM"
                                : m.type ===
                                  "hybrid"
                                ? "HYBRID"
                                : "CLASSICAL"}
                            </span>
                          </td>

                          <td
                            style={{
                              ...td,
                              color: C.hi,
                            }}
                          >
                            {m.accuracy !=
                            null
                              ? `${Number(
                                  m.accuracy
                                ).toFixed(
                                  2
                                )}%`
                              : "—"}
                          </td>

                          <td
                            style={{
                              ...td,
                              color: C.hi,
                            }}
                          >
                            {m.sensitivity !=
                            null
                              ? `${Number(
                                  m.sensitivity
                                ).toFixed(
                                  2
                                )}%`
                              : "—"}
                          </td>

                          <td
                            style={{
                              ...td,
                              color: C.hi,
                            }}
                          >
                            {m.specificity !=
                            null
                              ? `${Number(
                                  m.specificity
                                ).toFixed(
                                  2
                                )}%`
                              : "—"}
                          </td>

                          <td
                            style={{
                              ...td,
                              color: C.hi,
                            }}
                          >
                            {m.precision !=
                            null
                              ? `${Number(
                                  m.precision
                                ).toFixed(
                                  2
                                )}%`
                              : "—"}
                          </td>

                          <td
                            style={{
                              ...td,
                              color: C.hi,
                            }}
                          >
                            {m.f1 !=
                            null
                              ? `${Number(
                                  m.f1
                                ).toFixed(
                                  2
                                )}%`
                              : "—"}
                          </td>

                          <td
                            style={{
                              ...td,
                              color: C.hi,
                            }}
                          >
                            {m.roc_auc !=
                            null
                              ? `${Number(
                                  m.roc_auc
                                ).toFixed(
                                  2
                                )}%`
                              : "—"}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </Panel>
          </>
        )}

      <button
        onClick={reload}
        className="font-body"
        style={{
          marginTop: 16,
          background:
            "transparent",
          color: C.mid,
          border: `1px solid ${C.line}`,
          borderRadius: 9,
          padding: "9px 14px",
          fontSize: 13,
          cursor: "pointer",
          display: "flex",
          alignItems:
            "center",
          gap: 6,
        }}
      >
        <RefreshCcw size={13} />
        Refresh
      </button>
    </Section>
  );
}

/* ============================================================
   CONFUSION MATRIX
   ============================================================ */

function ConfusionMatrix({
  matrix,
}) {
  if (!matrix) {
    return (
      <div
        style={{
          padding: 20,
          color: C.mid,
          fontSize: 13,
        }}
      >
        No confusion matrix
        data available.
      </div>
    );
  }

  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;

  if (
    Array.isArray(matrix) &&
    matrix.length === 2 &&
    Array.isArray(matrix[0]) &&
    Array.isArray(matrix[1])
  ) {
    tn = matrix[0][0] ?? 0;
    fp = matrix[0][1] ?? 0;
    fn = matrix[1][0] ?? 0;
    tp = matrix[1][1] ?? 0;
  } else {
    tn =
      matrix.true_negative ??
      0;
    fp =
      matrix.false_positive ??
      0;
    fn =
      matrix.false_negative ??
      0;
    tp =
      matrix.true_positive ??
      0;
  }

  const cells = [
    {
      value: tn,
      label: "True Negative",
      tone: C.good,
    },
    {
      value: fp,
      label: "False Positive",
      tone: C.amber,
    },
    {
      value: fn,
      label: "False Negative",
      tone: C.risk,
    },
    {
      value: tp,
      label: "True Positive",
      tone: C.good,
    },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns:
          "repeat(2, 1fr)",
        gap: 10,
      }}
    >
      {cells.map(
        (cell) => (
          <div
            key={cell.label}
            style={{
              background:
                C.panel2,
              border: `1px solid ${C.line}`,
              borderRadius: 10,
              padding: 16,
              textAlign:
                "center",
            }}
          >
            <div
              className="font-display"
              style={{
                fontSize: 28,
                color:
                  cell.tone,
                fontWeight: 600,
              }}
            >
              {cell.value}
            </div>

            <div
              className="font-mono"
              style={{
                fontSize: 9.5,
                color: C.low,
                marginTop: 4,
              }}
            >
              {cell.label.toUpperCase()}
            </div>
          </div>
        )
      )}
    </div>
  );
}

/* ============================================================
   EVALUATION
   ============================================================ */
function Evaluation() {
  const {
    status,
    data,
    error,
    reload,
  } = useApiResource(
    getEvaluation,
    []
  );

  const [
    selectedDisease,
    setSelectedDisease,
  ] = useState(
    "heart_disease"
  );

  const [
    selectedModel,
    setSelectedModel,
  ] = useState(
    "classical"
  );

  const classicalData =
    status === "success" &&
    data
      ? data[selectedDisease]
      : null;

  const quantumData =
    status === "success" &&
    data?.quantum
      ? data.quantum[selectedDisease]
      : null;

  const hybridData =
    status === "success" &&
    data?.hybrid
      ? data.hybrid[selectedDisease]
      : null;

  let evaluationData = null;

  if (selectedModel === "classical") {
    evaluationData = classicalData;
  }

  if (selectedModel === "quantum") {
    evaluationData = quantumData;
  }

  if (selectedModel === "hybrid") {
    evaluationData = hybridData;
  }

  function formatMetric(value) {
    if (value == null) {
      return "—";
    }

    const numericValue = Number(value);

    if (Number.isNaN(numericValue)) {
      return "—";
    }

    const percentage =
      numericValue <= 1
        ? numericValue * 100
        : numericValue;

    return `${percentage.toFixed(2)}%`;
  }

  function handleDiseaseChange(e) {
    setSelectedDisease(
      e.target.value
    );

    setSelectedModel(
      "classical"
    );
  }

  return (
    <Section
      eyebrow="Model evaluation"
      title="Detailed evaluation metrics"
      color={C.teal}
      desc="Compare the returned classical, quantum and hybrid benchmark results."
    >
      {/* ================================================= */}
      {/* LOADING / ERROR */}
      {/* ================================================= */}

      {status !== "success" && (
        <StatusNotice
          status={status}
          error={error}
          loadingLabel="Loading evaluation..."
        />
      )}

      {/* ================================================= */}
      {/* NO EVALUATION DATA */}
      {/* ================================================= */}

      {status === "success" && !data && (
        <Panel>
          <span
            className="font-body"
            style={{
              color: C.mid,
              fontSize: 13.5,
            }}
          >
            No evaluation data returned yet.
          </span>
        </Panel>
      )}

      {/* ================================================= */}
      {/* EVALUATION CONTENT */}
      {/* ================================================= */}

      {status === "success" && data && (
        <>
          {/* ================================================= */}
          {/* SELECTORS */}
          {/* ================================================= */}

          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "space-between",
              gap: 16,
              marginBottom: 20,
              flexWrap: "wrap",
            }}
          >
            {/* DATASET SELECTOR */}

            <div>
              <div
                className="font-mono"
                style={{
                  color: C.mid,
                  fontSize: 10,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  marginBottom: 7,
                }}
              >
                Evaluation dataset
              </div>

              <select
                value={selectedDisease}
                onChange={handleDiseaseChange}
                className="font-body"
                style={{
                  background: C.panel,
                  color: C.hi,
                  border: `1px solid ${C.line}`,
                  borderRadius: 9,
                  padding: "10px 14px",
                  fontSize: 13,
                  minWidth: 220,
                  cursor: "pointer",
                }}
              >
                <option value="heart_disease">
                  UCI Heart Disease
                </option>

                <option value="breast_cancer">
                  Wisconsin Breast Cancer
                </option>
              </select>
            </div>

            {/* MODEL SELECTOR */}

            <div>
              <div
                className="font-mono"
                style={{
                  color: C.mid,
                  fontSize: 10,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  marginBottom: 7,
                }}
              >
                Evaluation model
              </div>

              <select
                value={selectedModel}
                onChange={(e) =>
                  setSelectedModel(
                    e.target.value
                  )
                }
                className="font-body"
                style={{
                  background: C.panel,
                  color: C.hi,
                  border: `1px solid ${C.line}`,
                  borderRadius: 9,
                  padding: "10px 14px",
                  fontSize: 13,
                  minWidth: 250,
                  cursor: "pointer",
                }}
              >
                <option value="classical">
                  Classical SVM
                </option>

                <option
                  value="quantum"
                  disabled={!quantumData}
                >
                  Quantum VQC
                </option>

                <option
                  value="hybrid"
                  disabled={!hybridData}
                >
                  Hybrid Quantum-Classical VQC
                </option>
              </select>
            </div>
          </div>

          {/* ================================================= */}
          {/* CURRENT MODEL */}
          {/* ================================================= */}

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 18,
            }}
          >
            <span
              className="font-body"
              style={{
                color: C.mid,
                fontSize: 12.5,
              }}
            >
              Currently viewing:
            </span>

            <span
              className="font-body"
              style={{
                color:
                  selectedModel === "hybrid"
                    ? C.violet
                    : selectedModel === "quantum"
                    ? C.teal
                    : C.good,
                fontSize: 13,
                fontWeight: 600,
              }}
            >
              {selectedModel === "classical"
                ? "Classical SVM"
                : selectedModel === "quantum"
                ? "Quantum VQC"
                : "Hybrid Quantum-Classical VQC"}
            </span>
          </div>

          {/* ================================================= */}
          {/* DATASET INFORMATION */}
          {/* ================================================= */}

          {evaluationData && (
            <Panel
              style={{
                marginBottom: 20,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 20,
                  flexWrap: "wrap",
                }}
              >
                {/* DATASET */}

                <div>
                  <div
                    className="font-mono"
                    style={{
                      color: C.mid,
                      fontSize: 10,
                      textTransform: "uppercase",
                      marginBottom: 7,
                    }}
                  >
                    Dataset
                  </div>

                  <div
                    className="font-body"
                    style={{
                      color: C.hi,
                      fontSize: 16,
                      fontWeight: 600,
                    }}
                  >
                    {evaluationData.dataset ?? "—"}
                  </div>
                </div>

                {/* MODEL */}

                <div>
                  <div
                    className="font-mono"
                    style={{
                      color: C.mid,
                      fontSize: 10,
                      textTransform: "uppercase",
                      marginBottom: 7,
                    }}
                  >
                    Model
                  </div>

                  <div
                    className="font-body"
                    style={{
                      color:
                        selectedModel === "hybrid"
                          ? C.violet
                          : selectedModel === "quantum"
                          ? C.teal
                          : C.good,
                      fontSize: 16,
                      fontWeight: 600,
                    }}
                  >
                    {evaluationData.model ?? "—"}
                  </div>
                </div>

                {/* DATASET ROWS */}

                <div>
                  <div
                    className="font-mono"
                    style={{
                      color: C.mid,
                      fontSize: 10,
                      textTransform: "uppercase",
                      marginBottom: 7,
                    }}
                  >
                    Dataset rows
                  </div>

                  <div
                    className="font-body"
                    style={{
                      color: C.hi,
                      fontSize: 16,
                      fontWeight: 600,
                    }}
                  >
                    {evaluationData.dataset_rows ?? "—"}
                  </div>
                </div>

                {/* EVALUATED ROWS */}

                <div>
                  <div
                    className="font-mono"
                    style={{
                      color: C.mid,
                      fontSize: 10,
                      textTransform: "uppercase",
                      marginBottom: 7,
                    }}
                  >
                    Evaluated rows
                  </div>

                  <div
                    className="font-body"
                    style={{
                      color: C.hi,
                      fontSize: 16,
                      fontWeight: 600,
                    }}
                  >
                    {evaluationData.evaluated_rows ?? "—"}
                  </div>
                </div>
              </div>
            </Panel>
          )}

          {/* ================================================= */}
          {/* METRICS */}
          {/* ================================================= */}

          {evaluationData && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 16,
              }}
              className="stat-grid"
            >
              <StatTile
                label="Accuracy"
                value={formatMetric(
                  evaluationData.accuracy
                )}
                tone={C.teal}
              />

              <StatTile
                label="Sensitivity"
                value={formatMetric(
                  evaluationData.sensitivity
                )}
                tone={C.good}
              />

              <StatTile
                label="Specificity"
                value={formatMetric(
                  evaluationData.specificity
                )}
                tone={C.good}
              />

              <StatTile
                label="Precision"
                value={formatMetric(
                  evaluationData.precision
                )}
                tone={C.amber}
              />

              <StatTile
                label="F1 Score"
                value={formatMetric(
                  evaluationData.f1 ??
                    evaluationData.f1_score
                )}
                tone={C.amber}
              />

              <StatTile
                label="ROC-AUC"
                value={formatMetric(
                  evaluationData.roc_auc
                )}
                tone={C.violet}
              />
            </div>
          )}

          {/* ================================================= */}
          {/* CONFUSION MATRIX IMAGE */}
          {/* ================================================= */}

          {evaluationData && (
            <Panel
              style={{
                marginTop: 20,
              }}
            >
              <PanelHeader
                label="Confusion matrix"
                sub={`Model: ${
                  evaluationData.model ?? "—"
                }`}
              />

              <ConfusionMatrixImage
                disease={selectedDisease}
                model={selectedModel}
              />
            </Panel>
          )}

          {/* ================================================= */}
          {/* NO MODEL DATA */}
          {/* ================================================= */}

          {!evaluationData && (
            <Panel>
              <span
                className="font-body"
                style={{
                  color: C.mid,
                  fontSize: 13.5,
                }}
              >
                Evaluation results for this model
                are not available yet.
              </span>
            </Panel>
          )}
        </>
      )}

      {/* ================================================= */}
      {/* REFRESH */}
      {/* ================================================= */}

      <button
        onClick={reload}
        className="font-body"
        style={{
          marginTop: 16,
          background: "transparent",
          color: C.mid,
          border: `1px solid ${C.line}`,
          borderRadius: 9,
          padding: "9px 14px",
          fontSize: 13,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <RefreshCcw size={13} />
        Refresh
      </button>
    </Section>
  );
}
function ConfusionMatrixImage({ disease, model }) {
  const imageMap = {
    heart_disease: {
      classical:
        "/confusion_matrices/Classical_model_heart.png",

      quantum:
        "/confusion_matrices/VQC_Heart_confusion_matrix.png",

      hybrid:
        "/confusion_matrices/hybrid_confusion_matrix.png",
    },

    breast_cancer: {
      classical:
        "/confusion_matrices/Classical_model_breast_cancer.png",

      quantum:
        "/confusion_matrices/VQC_BCancer_confusion_matrix.png",

      hybrid:
        "/confusion_matrices/wdbc_hybrid_confusion_matrix.png",
    },
  };

  const imageSrc =
    imageMap[disease]?.[model];

  if (!imageSrc) {
    return (
      <div
        className="font-body"
        style={{
          padding: 30,
          textAlign: "center",
          color: C.mid,
        }}
      >
        Confusion matrix image is not available.
      </div>
    );
  }

  return (
    <div
      style={{
        marginTop: 18,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
      }}
    >
      <img
        src={imageSrc}
        alt={`${disease} ${model} confusion matrix`}
        style={{
          display: "block",
          width: "100%",
          maxWidth: 900,
          height: "auto",
          borderRadius: 10,
          border: `1px solid ${C.line}`,
          background: C.panel,
        }}
      />
    </div>
  );
}
/* ============================================================
   DELIVERABLES
   ============================================================ */

function Deliverables() {
  return (
    <Section
      eyebrow="Scope"
      title="Delivery table"
      color={C.teal}
      desc="The complete CoherenceDx platform from preprocessing through quantum inference and evaluation."
    >
      <Panel
        style={{
          padding: 0,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            overflowX: "auto",
          }}
        >
          <table
            className="font-body"
            style={{
              width: "100%",
              borderCollapse:
                "collapse",
              minWidth: 640,
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom:
                    `1px solid ${C.line}`,
                  textAlign: "left",
                }}
              >
                <th style={th}>
                  #
                </th>

                <th style={th}>
                  Deliverable
                </th>

                <th style={th}>
                  Description
                </th>
              </tr>
            </thead>

            <tbody>
              {deliverables.map(
                (
                  [
                    n,
                    title,
                    desc,
                  ],
                  i
                ) => (
                  <tr
                    key={n}
                    style={{
                      borderBottom:
                        i <
                        deliverables.length -
                          1
                          ? `1px solid ${C.lineSoft}`
                          : "none",
                    }}
                  >
                    <td
                      style={{
                        ...td,
                        color: C.low,
                      }}
                      className="font-mono"
                    >
                      {n}
                    </td>

                    <td
                      style={{
                        ...td,
                        color: C.hi,
                        fontWeight: 600,
                        minWidth: 220,
                      }}
                    >
                      {title}
                    </td>

                    <td
                      style={{
                        ...td,
                        color: C.mid,
                      }}
                    >
                      {desc}
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </Section>
  );
}

/* ============================================================
   TABLE STYLES
   ============================================================ */

const th = {
  padding: "13px 20px",
  fontSize: 11.5,
  color: C.mid,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  fontWeight: 500,
};

const td = {
  padding: "14px 20px",
  fontSize: 13.3,
  lineHeight: 1.5,
  verticalAlign: "top",
};

/* ============================================================
   MAIN APP
   ============================================================ */

export default function App() {
  const [active, setActive] =
    useState("overview");

  return (
    <div
      style={{
        background: C.bg,
        minHeight: "100vh",
      }}
    >
      <style>
        {fontStyles}
      </style>

      <style>{`
        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          background: ${C.bg};
        }

        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }

        input[type=number] {
          -moz-appearance: textfield;
        }

        select,
        input,
        button {
          font-family: inherit;
        }

        .spin {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from {
            transform: rotate(0deg);
          }

          to {
            transform: rotate(360deg);
          }
        }

        @media (max-width: 820px) {
          .hero-grid {
            grid-template-columns: 1fr !important;
          }

          .pipeline-grid {
            grid-template-columns: 1fr 1fr !important;
          }

          .responsive-2col {
            grid-template-columns: 1fr !important;
          }

          .stat-grid {
            grid-template-columns: 1fr 1fr !important;
          }

          .feature-grid {
            grid-template-columns: 1fr !important;
          }
        }

        @media (max-width: 520px) {
          .pipeline-grid {
            grid-template-columns: 1fr !important;
          }

          .stat-grid {
            grid-template-columns: 1fr !important;
          }
        }

        button:focus-visible,
        a:focus-visible,
        input:focus-visible,
        select:focus-visible {
          outline: 2px solid ${C.teal};
          outline-offset: 2px;
        }
      `}</style>

      <Nav
        active={active}
        setActive={setActive}
      />

      {active ===
        "overview" && (
        <Overview
          setActive={setActive}
        />
      )}

      {active ===
        "prediction" && (
        <Prediction />
      )}

      {active ===
        "explain" && (
        <Explain />
      )}

      {active ===
        "benchmarks" && (
        <Benchmarks />
      )}

      {active ===
        "evaluation" && (
        <Evaluation />
      )}

      {active ===
        "deliver" && (
        <Deliverables />
      )}

      <div
        style={{
          borderTop: `1px solid ${C.line}`,
          padding: "26px 28px",
        }}
      >
        <div
          style={{
            maxWidth: 1180,
            margin: "0 auto",
            display: "flex",
            justifyContent:
              "space-between",
            flexWrap: "wrap",
            gap: 10,
          }}
        >
          <span
            className="font-mono"
            style={{
              color: C.low,
              fontSize: 11.5,
            }}
          >
            CoherenceDx · Hybrid
            QML for Early Disease
            Detection
          </span>

          <span
            className="font-mono"
            style={{
              color: C.low,
              fontSize: 11.5,
            }}
          >
            Research build · not for
            clinical use
          </span>
        </div>
      </div>
    </div>
  );
}