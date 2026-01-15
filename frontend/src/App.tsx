import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Search,
  Cpu,
  Layers,
  GitBranch,
  Database,
  Zap,
  ArrowRight,
  RefreshCcw,
  Info,
} from "lucide-react";
import type { Token, Expert, InferenceStep } from "../types";
import TopKProbabilities, {
  TokenNode,
  ExpertNode,
  ConnectionLines,
  TokenDetailPanel,
  AttentionHeatmap,
  MLAStats,
} from "./components/Visualizer";
import { generatePrompt } from "../services/generatePrompt";

const EXPERTS: Expert[] = [
  { id: 0, label: "Linguistic", load: 0, active: false },
  { id: 1, label: "Logical", load: 0, active: false },
  { id: 2, label: "Knowledge", load: 0, active: false },
  { id: 3, label: "Code", load: 0, active: false },
  { id: 4, label: "Reasoning", load: 0, active: false },
  { id: 5, label: "Safety", load: 0, active: false },
];

const App: React.FC = () => {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [tokens, setTokens] = useState<Token[]>([]);
  const [selectedToken, setSelectedToken] = useState<Token | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(-1);
  const [experts, setExperts] = useState<Expert[]>(EXPERTS);
  const [insights, setInsights] = useState<string>("");
  const [history, setHistory] = useState<{ in: string; out: string }[]>([]);
  const [kvCacheStatus, setKvCacheStatus] = useState<
    "idle" | "compressing" | "ready"
  >("idle");

  const candidates = [
    { token: "inference", prob: 88.5, color: "bg-cyan-500" },
    { token: "calculation", prob: 5.2, color: "bg-slate-600" },
    { token: "processing", prob: 3.1, color: "bg-slate-700" },
    { token: "analysis", prob: 1.8, color: "bg-slate-800" },
  ];

  const handleProcess = async () => {
    if (!input.trim() || isProcessing) return;

    setIsProcessing(true);
    setTokens([]);
    setSelectedToken(null);
    setOutput("");
    setCurrentStep(0);
    setInsights("Initializing Transformer Context...");
    setKvCacheStatus("idle");

    // Simulate Step 1: Tokenization & Embedding
    const words = input.toLowerCase().split(/\s+/);
    const mockTokens: Token[] = words.map((w, i) => ({
      id: i,
      text: w,
      embedding: Array.from({ length: 12 }, () => Math.random()),
      position: i,
      // Initial mock attention - more focused on itself
      attentionScores: words.map((_, idx) =>
        idx === i ? 0.6 + Math.random() * 0.4 : Math.random() * 0.2
      ),
    }));

    // Normalize mock attention scores
    mockTokens.forEach((t) => {
      const sum = (t.attentionScores || []).reduce((a, b) => a + b, 0);
      t.attentionScores = (t.attentionScores || []).map((s) => s / sum);
    });

    await new Promise((r) => setTimeout(r, 800));
    setTokens(mockTokens);
    setCurrentStep(1);
    // setInsights(await generatePrompt(input, "Tokenization & Embedding"));
    setInsights("");

    // Step 2: Attention (MLA)
    await new Promise((r) => setTimeout(r, 1200));
    setCurrentStep(2);
    // setInsights(
    //   await generatePrompt(input, "Multi-Head Latent Attention (MLA)")
    // );
    setInsights("");
    setKvCacheStatus("ready");

    // Step 3: DeepSeekMoE Routing
    await new Promise((r) => setTimeout(r, 1000));
    setCurrentStep(3);
    setExperts((prev) =>
      prev.map((e) => ({
        ...e,
        active: Math.random() > 0.6,
        load: Math.floor(Math.random() * 100),
      }))
    );
    // setInsights(
    //   await generatePrompt(input, "Mixture of Experts (MoE) Routing")
    // );
    setInsights("");

    // Step 4: Output Generation
    await new Promise((r) => setTimeout(r, 800));
    setCurrentStep(4);

    let finalOutput =
      "I'm processing your request. DeepSeek architecture is optimized for inference.";
    if (input.toLowerCase().includes("hello ai")) {
      finalOutput = "hallo im good how yours day? you good?";
    }

    setOutput(finalOutput);
    setHistory((prev) =>
      [{ in: input, out: finalOutput }, ...prev].slice(0, 5)
    );
    setIsProcessing(false);
    setInsights(
      "Inference Complete. Note how MLA and MoE reduced the computational overhead."
    );
  };

  const getStepColor = (step: number) => {
    if (currentStep < step) return "text-slate-600 border-slate-800";
    if (currentStep === step)
      return "text-cyan-400 border-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.2)]";
    return "text-green-400 border-green-900/40";
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col font-sans overflow-x-hidden">
      {/* --- HEADER --- */}
      {/* Perbaikan: Menggunakan w-full dan px-6 sesuai instruksi */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
        <div className="w-full px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-linear-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(34,211,238,0.4)]">
              <Cpu className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-cyan-400 to-purple-400">
                DeepSeek Explorer
              </h1>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-mono mt-0.5">
                Transformer From Scratch v3
              </p>
            </div>
          </div>

          <div className="flex gap-4 text-xs font-mono">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-700 shadow-inner">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
              <span className="text-slate-300">LIVE_INFERENCE</span>
            </div>
          </div>
        </div>
      </header>

      {/* --- MAIN CONTENT --- */}
      <main className="flex-1 w-full p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
        {/* KOLOM 1: Input & Pipeline Status (Kiri) */}
        <div className="lg:col-span-3 space-y-6 flex flex-col">
          {/* Section: Input */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
            <h2 className="text-[11px] font-bold uppercase text-slate-500 mb-4 flex items-center gap-2 tracking-wider">
              <Zap size={14} className="text-yellow-500" /> Inference Input
            </h2>
            <div className="relative group">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleProcess()}
                placeholder="Try 'jelaskan teori relativitas'..."
                className="w-full bg-slate-950 border border-slate-700 rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all font-mono text-sm text-slate-100 placeholder:text-slate-600"
              />
              <button
                onClick={handleProcess}
                disabled={isProcessing}
                className="absolute right-2 top-2 bottom-2 px-3 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-all shadow-lg hover:shadow-cyan-500/20 active:scale-95"
              >
                {isProcessing ? (
                  <RefreshCcw className="animate-spin" size={16} />
                ) : (
                  <ArrowRight size={16} />
                )}
              </button>
            </div>
          </section>

          {/* Section: Pipeline Steps */}
          <section className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5 flex-1">
            <h2 className="text-[11px] font-bold uppercase text-slate-500 mb-4 flex items-center gap-2 tracking-wider">
              <Layers size={14} className="text-purple-500" /> Pipeline Status
            </h2>
            <div className="space-y-3">
              {[
                { id: 1, label: "Tokenizer & Embedding", icon: Search },
                { id: 2, label: "Multi-Head Attention", icon: Zap },
                { id: 3, label: "Mixture of Experts (MoE)", icon: GitBranch },
                { id: 4, label: "Decoding Output", icon: Database },
              ].map((step) => (
                <div
                  key={step.id}
                  className={`flex items-center gap-3 p-3 rounded-xl border transition-all duration-300 ${getStepColor(
                    step.id
                  )}`}
                >
                  <div
                    className={`p-1.5 rounded-md ${
                      currentStep === step.id ? "bg-white/10" : "bg-transparent"
                    }`}
                  >
                    <step.icon size={16} />
                  </div>
                  <span className="text-xs font-semibold tracking-wide">
                    {step.label}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* KOLOM 2: Main Visualizer Area (Tengah) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* KOLOM 2: Main Visualizer Area (Tengah) */}
          <div className="lg:col-span-6 flex flex-col gap-6">
            {/* Main Canvas Area */}
            <div className="flex-1 bg-slate-900/30 border border-slate-800 rounded-3xl p-6 relative overflow-hidden flex flex-col items-center justify-center gap-10 min-h-175 shadow-2xl backdrop-blur-sm">
              {/* --- BACKGROUND LAYERS --- */}

              {/* Kita bungkus semua background di sini agar tidak terpengaruh Flexbox parent */}
              <div className="absolute inset-0 z-0">
                {/* Layer 1: Base Color */}
                <div className="absolute inset-0 bg-slate-950"></div>

                {/* Layer 2: GRID SYSTEM (Dibuat Presisi) */}
                {/* Menggunakan background-size 60px 60px yang strict agar kotak sempurna */}
                <div
                  className="absolute inset-0 pointer-events-none opacity-20"
                  style={{
                    backgroundImage: `
                            linear-gradient(to right, rgba(120, 120, 120, 0.1) 1px, transparent 1px),
                            linear-gradient(to bottom, rgba(120, 120, 120, 0.1) 1px, transparent 1px)
                        `,
                    backgroundSize: "60px 60px",
                  }}
                ></div>

                {/* Layer 3: Radial Glow (Pusat Cahaya) */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-200 h-200 bg-cyan-500/10 blur-[100px] rounded-full pointer-events-none"></div>

                {/* Layer 4: Vignette (Bayangan Tepi) */}
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(2,6,23,0.8)_100%)] pointer-events-none"></div>
              </div>

              {/* --- CONTENT AREA (z-10 agar di atas background) --- */}

              {/* Connection Lines */}
              <div className="absolute inset-0 pointer-events-none">
                <ConnectionLines active={currentStep >= 2} />
              </div>

              {/* Layer 1: Token Embeddings (Rata Tengah) */}
              <div className="z-10 w-full text-center mt-4 px-4">
                <h3 className="text-[10px] font-mono text-slate-500 mb-6 uppercase tracking-[0.3em] flex items-center justify-center gap-2">
                  <span className="w-8 h-px bg-slate-800"></span>
                  Input Embeddings
                  <span className="w-8 h-px bg-slate-800"></span>
                </h3>
                <div className="flex flex-wrap justify-center gap-3 max-w-3xl mx-auto">
                  {tokens.map((t: any) => (
                    <TokenNode
                      key={t.id}
                      token={t}
                      isActive={currentStep >= 1}
                      isSelected={selectedToken?.id === t.id}
                      onClick={() => setSelectedToken(t)}
                    />
                  ))}
                </div>
              </div>

              {/* Layer 2: Attention & Core Engine (Rata Tengah) */}
              <div
                className={`z-10 transition-all duration-1000 transform ${
                  currentStep >= 2
                    ? "opacity-100 scale-100 translate-y-0"
                    : "opacity-10 scale-90 translate-y-4 blur-sm"
                }`}
              >
                <div className="relative group">
                  <div className="absolute -inset-1 bg-linear-to-r from-cyan-500 to-blue-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-1000"></div>
                  <div className="relative px-8 py-4 rounded-2xl bg-slate-900 border border-slate-700/50 text-cyan-400 font-mono text-xs uppercase shadow-2xl flex flex-col items-center gap-2 min-w-50">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-2 h-2 rounded-full bg-cyan-400 ${
                          currentStep === 2 ? "animate-ping" : ""
                        }`}
                      ></div>
                      <span className="font-bold tracking-wider text-slate-100">
                        MLA Core Engine
                      </span>
                    </div>
                    <span className="text-[9px] text-slate-500 tracking-widest">
                      Multi-Head Latent Attention
                    </span>
                  </div>
                </div>
              </div>

              {/* Layer 3: Experts (MoE) - PERBAIKAN DI SINI */}
              <div
                className={`z-10 w-full max-w-5xl px-4 transition-all duration-700 ${
                  currentStep >= 3
                    ? "opacity-100 translate-y-0"
                    : "opacity-10 translate-y-8"
                }`}
              >
                <h3 className="text-xs font-mono text-slate-400 mb-6 text-center uppercase tracking-[0.3em] font-bold">
                  Active Experts Router
                </h3>

                {/* PERUBAHAN UTAMA: Gunakan flex + justify-center agar item sisa di baris bawah tetap di tengah */}
                <div className="flex flex-wrap justify-center gap-3 md:gap-4 lg:gap-6 max-w-3xl mx-auto w-full">
                  {experts.slice(0, 8).map((e: any) => (
                    // Tambahkan width fix (misal w-32 atau w-40) agar terlihat seperti grid tapi rata tengah
                    <div
                      key={e.id}
                      className="w-35 md:w-40 min-h-22.5 flex flex-col transform transition-all hover:scale-105"
                    >
                      <ExpertNode expert={e} />
                    </div>
                  ))}
                </div>
              </div>

              {/* Layer 4: Output Bubble */}
              <div
                className={`z-20 w-full pb-8 px-4 transition-all duration-500 ${
                  output
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-8"
                }`}
              >
                <div className="mx-auto max-w-2xl bg-slate-950/80 backdrop-blur-md border border-purple-500/30 rounded-2xl p-6 shadow-[0_0_50px_-10px_rgba(168,85,247,0.15)]">
                  <div className="flex flex-col items-center text-center gap-3">
                    <span className="text-[9px] uppercase tracking-widest text-purple-400">
                      Final Decoding
                    </span>
                    <p className="text-xl md:text-2xl font-medium text-transparent bg-clip-text bg-linear-to-r from-slate-100 to-slate-400 italic font-serif">
                      "{output}"
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Terminal / History Log */}
            <section className="bg-black/40 border border-slate-800/50 rounded-2xl overflow-hidden flex flex-col h-40 shadow-inner">
              <div className="px-4 py-2 border-b border-slate-800/50 bg-slate-900/30 flex justify-between items-center backdrop-blur-sm">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50"></div>
                  </div>
                  <h2 className="text-[10px] text-slate-400 font-mono uppercase tracking-widest ml-2">
                    System_Logs
                  </h2>
                </div>
                <span className="text-[9px] text-cyan-500 font-mono bg-cyan-950/20 px-2 py-0.5 rounded border border-cyan-900/50">
                  v2.5.0-FLASH
                </span>
              </div>

              {/* CUSTOM SCROLLBAR SECTION */}
              <div
                className="p-3 overflow-y-auto font-mono text-[10px] space-y-1.5 
              [&::-webkit-scrollbar]:w-1.5 
              [&::-webkit-scrollbar-track]:bg-transparent 
              [&::-webkit-scrollbar-thumb]:bg-slate-800 
              [&::-webkit-scrollbar-thumb]:rounded-full 
              hover:[&::-webkit-scrollbar-thumb]:bg-slate-700"
              >
                {history.length === 0 ? (
                  <div className="text-slate-600 italic pl-2 border-l-2 border-slate-800/50 py-1">
                    <span className="animate-pulse">_</span> Ready for
                    inference...
                  </div>
                ) : (
                  history.map((h: any, i: number) => (
                    <div
                      key={i}
                      className="group flex flex-col gap-0.5 p-1.5 rounded hover:bg-white/5 transition-colors border-l-2 border-transparent hover:border-cyan-500/50"
                    >
                      <div className="flex items-center gap-2 text-slate-600 text-[9px]">
                        <span>{new Date().toLocaleTimeString()}</span>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-cyan-500/80 font-bold">IN:</span>
                        <span className="text-slate-300">{h.in}</span>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-purple-500/80 font-bold">
                          OUT:
                        </span>
                        <span className="text-slate-300">{h.out}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>
          </div>
        </div>

        {/* KOLOM 3: Detail Panel & Heatmap (Kanan) */}
        <div className="lg:col-span-4 space-y-6 flex flex-col">
          {/* Panel: Token Inspector */}
          <div className="transition-all duration-500 min-h-75">
            {selectedToken ? (
              <TokenDetailPanel
                token={selectedToken}
                allTokens={tokens}
                onClose={() => setSelectedToken(null)}
                kvCacheStatus={kvCacheStatus}
              />
            ) : (
              <section className="h-full bg-slate-900/40 border border-slate-800 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center group hover:bg-slate-900/60 transition-colors">
                <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                  <Info
                    className="text-slate-600 group-hover:text-cyan-400 transition-colors"
                    size={24}
                  />
                </div>
                <h3 className="text-slate-400 text-sm font-bold uppercase tracking-wider mb-2">
                  Deep Inspection Mode
                </h3>
                <p className="text-slate-500 text-xs leading-relaxed max-w-62.5">
                  Klik salah satu token di visualizer tengah untuk melihat
                  analisis KV Cache dan vektor attention secara detail.
                </p>
              </section>
            )}
          </div>

          {/* Panel: Next Token Prediction */}
          <TopKProbabilities input={input} />

          {/* Memory Usage */}
          <MLAStats tokens={tokens} />

          {/* Panel: Mini Logs / Footer Status */}
          <section className="bg-black/40 border border-slate-800 rounded-2xl p-4 font-mono text-[9px]">
            <div className="flex justify-between border-b border-slate-800 pb-2 mb-2 text-slate-400">
              <span>SYSTEM_DIAGNOSTICS</span>
              <span className="text-green-500 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>{" "}
                ONLINE
              </span>
            </div>
            <div className="space-y-1 text-slate-500">
              <p>{`> Initializing DeepSeek-V3 architecture... OK`}</p>
              <p>{`> Loading MoE Router Config... OK`}</p>
              {tokens.length > 0 && (
                <p className="text-cyan-500 animate-pulse">
                  {`> Processing batch: ${tokens.length} tokens active.`}
                </p>
              )}
            </div>
          </section>
        </div>
      </main>

      {/* --- FOOTER --- */}
      <footer className="py-6 text-center border-t border-slate-900 bg-slate-950">
        <p className="text-slate-700 text-[9px] uppercase tracking-[0.3em] font-mono">
          Engineered with React & Tailwind • Multi-Head Latent Attention
          Simulator
        </p>
      </footer>
    </div>
  );
};

export default App;
