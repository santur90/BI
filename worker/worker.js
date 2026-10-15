export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "https://santurlabs.site",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };
    const json = (obj, status = 200) =>
      new Response(JSON.stringify(obj), {
        status, headers: { ...cors, "Content-Type": "application/json" }
      });

    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);

    // 诊断 1：binding 是否存在
    if (!env.AI) return json({ error: "env.AI is undefined", hint: "binding 名称不是 AI 或未生效" }, 500);

    let body;
    try {
      body = await request.json();
    } catch (e) {
      return json({ error: "Bad JSON", detail: String(e) }, 400);
    }

    const module = String(body.module || "bioinformatics");
    const context = String(body.context || "").slice(0, 6000);
    const question = String(body.question || "Interpret these results.").slice(0, 500);

    const messages = [
      {
        role: "system",
        content:
          `You are a bioinformatics assistant interpreting a ${module} analysis. ` +
          "Interpret ONLY the summary statistics provided. Never invent genes, p-values, or effect sizes. " +
          "State association, not causation. Flag small sample sizes or multiple-testing concerns. " +
          "Output: 3-5 bullet points, then 1 suggested next analysis."
      },
      { role: "user", content: `Result summary:\n${context}\n\nQuestion: ${question}` }
    ];

    try {
        const result = await env.AI.run("@cf/google/gemma-4-26b-a4b-it", {
          messages, max_tokens: 800
        });

        const answer =
          result?.choices?.[0]?.message?.content ||
          result?.response ||
          null;

        if (!answer) {
          return json({ error: "Empty answer from model", finish_reason: result?.choices?.[0]?.finish_reason || null }, 502);
        }
        return json({ answer: answer.trim() });
    } catch (e) {
      console.error("AI error:", e);
      // 诊断 2：返回真实错误
      return json({
        error: "AI request failed",
        detail: String(e?.message || e),
        name: e?.name || null
      }, 500);
    }
  }
};