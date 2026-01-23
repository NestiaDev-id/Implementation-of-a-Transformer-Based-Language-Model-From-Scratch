import React from "react";
import type { Token, Expert } from "../../types";
import { Info, X } from "lucide-react";
import { useState, useMemo, useEffect } from "react";

export const TokenNode: React.FC<{
  token: Token;
  isActive: boolean;
  isSelected: boolean;
  onClick: () => void;
}> = ({ token, isActive, isSelected, onClick }) => (
  <button
    onClick={onClick}
    className={`
      p-3 rounded-lg border transition-all duration-300 transform text-center justify-center
      ${
        isSelected
          ? "bg-cyan-500/40 border-cyan-300 scale-110 shadow-[0_0_20px_rgba(34,211,238,0.5)] z-20"
          : isActive
            ? "bg-cyan-500/10 border-cyan-500/50 hover:border-cyan-400"
            : "bg-slate-900 border-slate-700 opacity-60"
      }
    `}
  >
    <div className="text-[10px] text-slate-400 mb-1 font-mono uppercase">
      {/* IDX adalah posisi token */}
      IDX: {token.position}
    </div>
    <div className="font-mono text-sm font-bold truncate max-w-20">
      {token.text}
    </div>
  </button>
);

export const AttentionHeatmap: React.FC<{
  tokens: Token[];
}> = ({ tokens }) => {
  const [hoveredCell, setHoveredCell] = useState<{
    r: number;
    c: number;
  } | null>(null);

  const topAttentions = useMemo(() => {
    return tokens
      .map((token) => {
        const scores = token.attentionScores || [];
        const topIdx = scores.indexOf(Math.max(...scores));
        return {
          source: token.text,
          target: tokens[topIdx]?.text || "N/A",
          weight: scores[topIdx] || 0,
        };
      })
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 3);
  }, [tokens]);

  if (tokens.length === 0) return null;

  return (
    <div className="w-full bg-slate-900/50 border border-slate-800 rounded-2xl p-6 overflow-hidden flex flex-col items-center animate-in fade-in duration-700">
      <div className="w-full flex justify-between items-center mb-6">
        <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-widest flex items-center gap-2">
          Attention Matrix (MLA Heatmap)
        </h3>
        <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
          <span>0.0</span>
          <div className="w-24 h-2 bg-linear-to-r from-slate-900 via-cyan-900 to-cyan-400 rounded-full border border-slate-700"></div>
          <span>1.0</span>
        </div>
      </div>

      <div className="relative overflow-auto max-w-full custom-scrollbar pb-4">
        <div
          className="grid gap-1"
          style={{
            gridTemplateColumns: `auto repeat(${tokens.length}, minmax(32px, 1fr))`,
            minWidth: `${tokens.length * 36 + 80}px`,
          }}
        >
          {/* Header row (Target Tokens) */}
          <div className="h-10"></div>
          {tokens.map((t, i) => (
            <div
              key={`h-${i}`}
              className="flex items-center justify-center p-1"
            >
              <span className="text-[9px] font-mono text-slate-400 rotate-45 origin-bottom-left whitespace-nowrap">
                {t.text}
              </span>
            </div>
          ))}

          {/* Rows (Source Tokens) */}
          {tokens.map((source, rIdx) => (
            <React.Fragment key={`row-${rIdx}`}>
              <div className="flex items-center justify-end pr-3 h-8">
                <span className="text-[10px] font-mono text-slate-400 truncate max-w-15">
                  {source.text}
                </span>
              </div>
              {tokens.map((target, cIdx) => {
                const score = source.attentionScores
                  ? source.attentionScores[cIdx]
                  : 0;
                const isHovered =
                  hoveredCell?.r === rIdx && hoveredCell?.c === cIdx;
                const isRelated =
                  hoveredCell?.r === rIdx || hoveredCell?.c === cIdx;

                return (
                  <div
                    key={`cell-${rIdx}-${cIdx}`}
                    onMouseEnter={() => {
                      setHoveredCell({ r: rIdx, c: cIdx });
                    }}
                    onMouseLeave={() => {
                      setHoveredCell(null);
                    }}
                    className={`h-8 w-full rounded-sm transition-all duration-150 cursor-crosshair relative group ${
                      isHovered
                        ? "ring-2 ring-white z-10 scale-110"
                        : isRelated
                          ? "ring-1 ring-white/10"
                          : ""
                    }`}
                    style={{
                      backgroundColor: `rgba(34, 211, 238, ${
                        0.05 + score * 0.95
                      })`,
                      opacity: isRelated ? 1 : 0.8,
                    }}
                  >
                    {isHovered && (
                      <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-slate-950 border border-cyan-500/50 rounded px-2 py-1 text-[9px] font-mono text-cyan-400 whitespace-nowrap shadow-xl z-50">
                        {source.text} → {target.text}: {score.toFixed(3)}
                      </div>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="mt-4 flex items-start gap-2 bg-cyan-950/20 border border-cyan-900/40 p-3 rounded-lg w-full">
        <Info size={14} className="text-cyan-500 shrink-0 mt-0.5" />
        <p className="text-[10px] text-slate-400 leading-relaxed italic">
          The X-axis represents the{" "}
          <span className="text-cyan-400">Target (Key)</span> tokens, while the
          Y-axis represents the{" "}
          <span className="text-cyan-400">Source (Query)</span> tokens.
          Brightness indicates how strongly the model associates words in its
          latent projection.
        </p>
      </div>
    </div>
  );
};

// Perbarui interface props
export const TokenDetailPanel: React.FC<{
  token: Token;
  allTokens: Token[];
  onClose: () => void;
  kvCacheStatus: "idle" | "compressing" | "ready"; // Tambahkan prop ini
}> = ({ token, allTokens, onClose, kvCacheStatus }) => {
  // Terima prop ini

  // Helper function untuk menampilkan teks dan warna yang tepat
  const getCacheDisplay = () => {
    if (kvCacheStatus === "ready") {
      return {
        text: "Compressed KV Cache Ready (~15% Memory)",
        color: "text-green-400",
        barWidth: "w-[15%]",
      };
    }
    if (kvCacheStatus === "compressing") {
      return {
        text: "Compressing KV Cache (MLA)",
        color: "text-cyan-400",
        barWidth: "w-[50%]",
      };
    }
    return {
      text: "KV Cache Status Idle",
      color: "text-slate-500",
      barWidth: "w-0",
    };
  };

  const cacheDisplay = getCacheDisplay();

  return (
    <div className="bg-slate-900 border border-cyan-500/30 rounded-2xl p-5 shadow-2xl animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-cyan-400 font-bold flex items-center gap-2">
            Token Analysis:{" "}
            <span className="text-white font-mono">"{token.text}"</span>
          </h3>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest mt-1">
            Deep Dimensional inspection
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-slate-500 hover:text-white transition-colors"
        >
          <X size={18} />
        </button>
      </div>

      <div className="space-y-4">
        <div>
          <h4 className="text-[10px] font-bold text-slate-400 uppercase mb-2">
            Embedding Vector (Truncated)
          </h4>
          <div className="grid grid-cols-4 gap-1">
            {token.embedding.map((val, i) => (
              <div
                key={i}
                className="bg-slate-950 border border-slate-800 p-1 text-center font-mono text-[9px] text-cyan-500/80 rounded"
              >
                {val.toFixed(4)}
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-[10px] font-bold text-slate-400 uppercase mb-2">
            Self-Attention Map
          </h4>
          <div className="space-y-1 max-h-32 overflow-y-auto pr-2 custom-scrollbar">
            {allTokens.map((t, idx) => {
              const score = token.attentionScores
                ? token.attentionScores[idx]
                : 0;
              return (
                <div
                  key={t.id}
                  className="flex items-center gap-2 text-[10px] font-mono"
                >
                  <span className="w-12 text-slate-500 truncate">{t.text}</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-linear-to-r from-cyan-600 to-purple-500 transition-all duration-1000"
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                  <span className="w-8 text-right text-slate-300">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <h4 className="text-[10px] font-bold text-slate-400 uppercase mb-2">
            KV Cache Status
          </h4>
          {/* Status Info */}
          <div className="space-y-1">
            <div className="flex justify-between text-[9px] font-mono">
              <span className="text-slate-500">
                Mode: Multi-Head Latent Attention
              </span>
              <span className={cacheDisplay.color}>{cacheDisplay.text}</span>
            </div>
            <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full bg-cyan-500 transition-all duration-1000 ${cacheDisplay.barWidth}`}
              />
            </div>
          </div>

          {/* Heatmap Call */}
          <div className="mt-2 opacity-90 hover:opacity-100 transition-opacity">
            <AttentionHeatmap tokens={allTokens} />
          </div>
        </div>
      </div>
    </div>
  );
};

export const MLAStats = ({ tokens }: { tokens: any[] }) => {
  // LOGIKA PERHITUNGAN MEMORI (SIMULASI)
  const stats = useMemo(() => {
    const tokenCount = tokens.length > 0 ? tokens.length : 0;

    // Asumsi: Standard FP16 memakan ~1.5 MB per token (hanya simulasi kasar untuk visualisasi)
    // Asumsi: MLA DeepSeek memakan ~0.09 MB per token (Kompresi ~93%)
    const standardUsage = tokenCount * 1.5;
    const mlaUsage = tokenCount * 0.09;

    // Hitung persentase kompresi
    // Jika token 0, default ke 93.5% (nilai teoritis paper)
    const compression =
      tokenCount > 0 ? (1 - mlaUsage / standardUsage) * 100 : 93.5;

    // Hitung lebar bar visual
    const barWidth = tokenCount > 0 ? (mlaUsage / standardUsage) * 100 : 6.5;

    return {
      mlaVal: mlaUsage.toFixed(2), // Tampilkan 2 desimal
      compVal: compression.toFixed(1),
      width: barWidth,
    };
  }, [tokens]);

  return (
    <section className="bg-linear-to-br from-slate-900 to-slate-950 border border-indigo-500/30 rounded-2xl p-4 relative overflow-hidden group shadow-lg">
      {/* Background Decor */}
      <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          className="text-indigo-500"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
          />
        </svg>
      </div>

      <h3 className="text-[10px] font-bold text-indigo-400 uppercase mb-3 tracking-widest flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></span>
        MLA Efficiency (KV Cache)
      </h3>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-[9px] text-slate-500 uppercase font-semibold">
            Memory Usage
          </p>
          <div className="flex items-baseline gap-1">
            <p className="text-xl font-mono text-slate-200 transition-all duration-300">
              {stats.mlaVal}
            </p>
            <span className="text-xs text-slate-600 font-mono">MB</span>
          </div>
        </div>
        <div>
          <p className="text-[9px] text-slate-500 uppercase font-semibold">
            Compression
          </p>
          <div className="flex items-baseline gap-1">
            <p className="text-xl font-mono text-indigo-400 transition-all duration-300">
              {stats.compVal}
            </p>
            <span className="text-xs text-indigo-600 font-mono">%</span>
          </div>
        </div>
      </div>

      {/* Visual Bar Comparison */}
      <div className="mt-4 w-full h-1.5 bg-slate-800 rounded-full flex overflow-hidden relative">
        {/* Bar MLA (Aktif) */}
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-500 ease-out shadow-[0_0_10px_rgba(99,102,241,0.5)]"
          style={{ width: `${stats.width}%` }}
        ></div>

        {/* Marker Standard (Hanya untuk perbandingan visual) */}
        <div className="absolute right-0 top-0 bottom-0 w-0.5 bg-red-500/30"></div>
      </div>

      <div className="flex justify-between mt-1.5">
        <p className="text-[8px] text-indigo-400 font-mono">MLA Active</p>
        <p className="text-[8px] text-slate-600 font-mono">
          Standard MHA Limit
        </p>
      </div>
    </section>
  );
};

export const TopKProbabilities = ({ input }: { input: string }) => {
  const [predictions, setPredictions] = useState<any[]>([]);

  // KAMUS PREDIKSI (Mini-Language Model Simulation)
  // Kiri: Kata terakhir user, Kanan: Kandidat kata selanjutnya
  const PREDICTION_DB: Record<string, string[]> = {
    hello: ["world", "ai", "there", "user"],
    deep: ["seek", "learning", "space", "dive"],
    deepseek: ["coder", "chat", "llm", "v3"],
    artificial: ["intelligence", "neural", "general", "life"],
    machine: ["learning", "vision", "translation", "code"],
    neural: ["network", "processing", "engine", "data"],
    large: ["language", "scale", "model", "data"],
    language: ["model", "processing", "generation", "understanding"],
    transformer: ["architecture", "decoder", "layer", "attention"],
    multi: ["head", "modal", "layer", "task"],
    attention: ["mechanism", "score", "mask", "head"],
    active: ["experts", "router", "param", "learning"],
    mixture: ["of-experts", "model", "strategy", "layer"],
    kv: ["cache", "compression", "memory", "store"],
  };

  // Fallback jika kata tidak dikenali (Default AI Terms)
  const FALLBACK_TOKENS = [
    "inference",
    "computation",
    "analysis",
    "output",
    "token",
    "probability",
  ];

  useEffect(() => {
    // 1. Ambil kata terakhir dari input
    const words = input.trim().split(/\s+/);
    const lastWord = words[words.length - 1]?.toLowerCase() || "";

    // 2. Cari kandidat (Lookup atau Fallback)
    let candidates = PREDICTION_DB[lastWord];

    // Jika tidak ada match, ambil 4 random dari fallback
    if (!candidates) {
      // Shuffle fallback array
      candidates = [...FALLBACK_TOKENS]
        .sort(() => 0.5 - Math.random())
        .slice(0, 4);
    }

    // 3. Generate Probabilitas (Logits Simulation)
    // Kandidat pertama selalu dapat probabilitas terbesar (60-90%)
    const topProb = Math.floor(Math.random() * (95 - 60 + 1) + 60);
    let remaining = 100 - topProb;

    const newPredictions = candidates.map((token, index) => {
      let prob;
      if (index === 0) {
        prob = topProb;
      } else if (index === candidates.length - 1) {
        prob = remaining; // Sisa terakhir
      } else {
        // Ambil potongan acak dari sisa
        const slice = Math.floor(Math.random() * (remaining - 5 + 1) + 5);
        // Pastikan tidak mengambil lebih dari yang tersedia (safety check sederhana)
        prob = slice > remaining ? remaining : slice;
        remaining -= prob;
      }

      return {
        token,
        prob: prob < 0 ? 0 : prob, // Safety no negative
        // Warna dinamis: Top 1 Cyan, sisanya makin gelap
        color:
          index === 0
            ? "bg-cyan-500"
            : index === 1
              ? "bg-cyan-600/80"
              : index === 2
                ? "bg-cyan-700/60"
                : "bg-cyan-800/40",
      };
    });

    setPredictions(newPredictions);
  }, [input]); // Re-run setiap kali input berubah

  return (
    <section className="bg-slate-900/40 border border-slate-800 rounded-2xl p-5 backdrop-blur-sm mt-6 transition-all duration-500">
      <h3 className="text-[10px] font-bold text-slate-500 uppercase mb-4 tracking-widest flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
        Next Token Prediction (Logits)
      </h3>

      <div className="space-y-3">
        {predictions.map((c, i) => (
          <div key={i} className="group relative">
            <div className="flex justify-between text-[10px] font-mono mb-1 text-slate-400 z-10 relative">
              <span
                className={`${
                  i === 0 ? "text-cyan-300 font-bold" : ""
                } flex items-center gap-2`}
              >
                <span className="opacity-30">#{i + 1}</span> "{c.token}"
              </span>
              <span>{c.prob.toFixed(1)}%</span>
            </div>

            {/* Progress Bar Container */}
            <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full ${c.color} rounded-full transition-all duration-700 ease-out`}
                style={{ width: `${c.prob}%` }}
              ></div>
            </div>

            {/* Hover Tooltip (Efek Visual Tambahan) */}
            <div className="absolute left-0 -top-full opacity-0 group-hover:opacity-100 transition-opacity bg-black/80 text-[9px] text-white px-2 py-1 rounded border border-slate-700 pointer-events-none transform -translate-y-1">
              Token ID: {Math.floor(Math.random() * 50000)}
            </div>
          </div>
        ))}
      </div>

      {/* Footer Info */}
      <div className="mt-4 pt-3 border-t border-slate-800/50 flex justify-between items-center text-[8px] text-slate-600">
        <span>Sampling: Greedy</span>
        <span>Temp: 0.7</span>
      </div>
    </section>
  );
};

export default TopKProbabilities;

export const ExpertNode: React.FC<{ expert: Expert }> = ({ expert }) => (
  <div
    className={`
    w-24 h-24 rounded-xl border flex flex-col items-center justify-center transition-all duration-500
    ${
      expert.active
        ? "bg-purple-500/30 border-purple-400 shadow-[0_0_10px_rgba(168,85,247,0.2)]"
        : "bg-slate-900 border-slate-800"
    }
  `}
  >
    <div className="text-[10px] uppercase text-slate-500 font-bold mb-1 tracking-tighter">
      Expert {expert.id}
    </div>
    <div className="text-[11px] text-center px-1 leading-tight font-medium">
      {expert.label}
    </div>
    <div className="mt-2 w-12 h-1 bg-slate-800 rounded-full overflow-hidden">
      <div
        className="h-full bg-purple-500 transition-all duration-1000"
        style={{ width: `${expert.load}%` }}
      />
    </div>
  </div>
);

export const ConnectionLines: React.FC<{ active: boolean }> = ({ active }) => (
  <svg className="absolute inset-0 w-full h-full pointer-events-none overflow-visible">
    <defs>
      <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#22d3ee" stopOpacity="0" />
        <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.5" />
        <stop offset="100%" stopColor="#a855f7" stopOpacity="0.8" />
      </linearGradient>
    </defs>
    {active && (
      <path
        d="M 100,50 Q 250,50 400,150 T 700,250"
        fill="none"
        stroke="url(#grad)"
        strokeWidth="2"
        className="flow-animation"
      />
    )}
  </svg>
);
