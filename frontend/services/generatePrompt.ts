export async function generatePrompt(prompt: string, stage: string) {
  try {
    // const response = await ai.models.generateContent({
    //   model: "gemini-3-flash-preview",
    //   contents: `Explain how a DeepSeek model specifically handles the ${stage} stage for the input: "${prompt}".
    //   Focus on concepts like Multi-Head Latent Attention (MLA) and DeepSeekMoE.
    //   Keep the tone professional and technical yet accessible for someone learning transformers from scratch.
    //   Maximum 3 sentences.`,
    // });
    // return response.text || "Insight unavailable.";
  } catch (error) {
    console.error("Gemini Insight Error:", error);
    return "Error fetching architectural insight.";
  }
}
