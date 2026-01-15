export interface Token {
  id: number;
  text: string;
  embedding: number[];
  position: number;
  attentionScores?: number[];
}

export interface AttentionHead {
  id: number;
  weights: number[][];
  active: boolean;
}

export interface Expert {
  id: number;
  load: number;
  label: string;
  active: boolean;
}

export type AttentionData = {
  tokens: Token[];
  heads: AttentionHead[];
};

export type MoEData = {
  tokens: Token[];
  experts: Expert[];
};

export type OutputData = {
  output: string;
  tokens: Token[];
};

export interface InferenceStep {
  stage: "Tokenization" | "Embedding" | "Attention" | "MoE" | "Output";
  description: string;
  data: Token[] | AttentionData | MoEData | OutputData;
}
