import os
import re
import json
import random
import markdown
import concurrent.futures
from functools import lru_cache
from flask import Flask, render_template, request
from dotenv import load_dotenv
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()
app = Flask(__name__)

# ----------------------------------------------------------------------
# PROMPTS – loaded from .env
# ----------------------------------------------------------------------
def load_prompts() -> dict:
    return {
        "CHIEF_SCOUT_AGENT": os.getenv("PROMPT_CHIEF_SCOUT_AGENT"),
        "GENERAL_PLANNER": os.getenv("PROMPT_GENERAL_PLANNER"),
        "TELEMEDICINE_PLANNER": os.getenv("PROMPT_TELEMEDICINE_PLANNER"),
        "GOOGLE_PIXEL_PLANNER": os.getenv("PROMPT_GOOGLE_PIXEL_PLANNER"),
        "SAIDLER_PLANNER": os.getenv("PROMPT_SAIDLER_PLANNER", os.getenv("PROMPT_GENERAL_PLANNER")),
        "UNIVERSAL_GOOGLE_ADS": os.getenv("PROMPT_UNIVERSAL_GOOGLE_ADS"),
        "VIRAL_X_POST": os.getenv("PROMPT_VIRAL_X_POST"),
    }

PROMPTS = load_prompts()
missing = [k for k, v in PROMPTS.items() if not v]
if missing:
    raise EnvironmentError(f"Missing prompts in .env: {missing}")

# ----------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------
def smart_truncate(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    last_space = text.rfind(" ", 0, length)
    return text[:length] if last_space == -1 else text[:last_space]

def extract_json_from_string(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"```(json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON error: {e} → {json_str[:100]}...")
        return None

# ----------------------------------------------------------------------
# MODEL CACHE
# ----------------------------------------------------------------------
MODEL_CACHE: dict = {}
def get_client() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    configure(api_key=key)

def select_model(kind: str) -> tuple[str, GenerativeModel]:
    name = "gemini-flash-lite-latest" if kind == "lite" else "gemini-2.5-pro"
    if name not in MODEL_CACHE:
        MODEL_CACHE[name] = GenerativeModel(name)
    return name, MODEL_CACHE[name]

# ----------------------------------------------------------------------
# AGENTS
# ----------------------------------------------------------------------
class ChiefScoutAgent:
    def strategize(self, goal: str) -> dict:
        prompt = PROMPTS["CHIEF_SCOUT_AGENT"] or ""
        prompt = prompt.format(TOPIC=goal)
        try:
            _, model = select_model("lite")
            resp = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            plan = extract_json_from_string(resp.text)
            if plan and all(k in plan for k in ("reasoning", "planner_tool", "domain_key")):
                # ---- SAIDLER & CO AUTO-DETECT (Expanded for 2025) ----
                saidler_kw = [
                    "saidler", "zug", "portfolio", "investment", "finance", "blockchain", "media",
                    "rag", "mcp", "fine-tune", "llm", "governance", "ethics", "bias", "compliance",
                    "due diligence", "tokenization", "rwa", "defi", "grok 4", "workshop", "dashboard",
                    "recommendation system", "automation workflow", "custom gpt", "trend scouting",
                    "mmcos", "async", "orchestration", "event-driven", "non-blocking", "human-in-loop",
                ]
                if any(kw in goal.lower() for kw in saidler_kw):
                    plan["planner_tool"] = "SAIDLER_PLANNER"
                    plan["domain_key"] = "saidler_co"
                    plan["reasoning"] += (
                        " | Specialized: Injecting 2025 AI-blockchain trends, RAG, ethical governance, MMCOS async scaling."
                    )
                # --- ADVANCED TOOL ROUTING (Job Role + MMCOS) ---
                tool_map = {
                    "briefing|leadership|slide|presentation|strategy": "SAIDLER_AI_STRATEGY",
                    "rag|mcp|retrieval|chunking|embedding|pipeline": "SAIDLER_RAG_BLUEPRINT",
                    "ethics|bias|governance|compliance|finma|transparency|playbook": "SAIDLER_ETHICS_FRAMEWORK",
                    "dashboard|radar|live|api|arxiv|hugging face|davos": "SAIDLER_TREND_DASHBOARD",
                    "workshop|training|adoption|hands-on|program": "SAIDLER_PLANNER",  # fallback + KPIs
                }
                for pattern, tool in tool_map.items():
                    if re.search(pattern, goal.lower()):
                        plan["planner_tool"] = tool
                        plan["domain_key"] = "saidler_co"
                        plan["reasoning"] += f" | Role Match: {tool.replace('SAIDLER_', '').replace('_', ' ')}"
                        break
                return plan
            raise ValueError("Invalid scout output")
        except Exception as e:
            print(f"Scout error: {e}")
            domain = "saidler_co" if any(kw in goal.lower() for kw in saidler_kw) else "general"
            return {
                "reasoning": "Auto-recovery: defaulting to general (or Saidler & Co specialized).",
                "planner_tool": "SAIDLER_PLANNER" if domain == "saidler_co" else "GENERAL_PLANNER",
                "domain_key": domain,
            }

class PlannerAgent:
    @lru_cache(maxsize=100)
    def plan(self, goal: str, trends: str, tool_key: str) -> dict:
        tmpl = PROMPTS.get(tool_key, PROMPTS["GENERAL_PLANNER"]) or ""
        prompt = tmpl.format(TOPIC=goal, TRENDS=trends)
        try:
            _, model = select_model("lite")
            resp = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            plan = extract_json_from_string(resp.text)
            if plan and "mission_workflow" in plan:
                if tool_key.startswith("SAIDLER_"):
                    plan["mission_workflow"]["kpis"] = [
                        "RAG Efficiency: 42% faster reports",
                        "Trend Accuracy: 95% via Grok 4",
                        "Compliance Score: 100% ethical prompts",
                        "Adoption Rate: 80% staff trained in 30 days",
                        "Async Uplift: 30% GPU savings via MMCOS",
                    ]
                return plan
            raise ValueError("Planner JSON missing mission_workflow")
        except Exception as e:
            print(f"Planner error: {e}")
            fallback = {"mission_workflow": {"steps": ["Fallback plan"]}}
            if tool_key.startswith("SAIDLER_"):
                fallback["mission_workflow"]["kpis"] = ["Fallback: Standard efficiency tracking"]
            return fallback

class OrchestratorAgent:
    def execute(self, plan: dict, goal: str, domain: str) -> dict:
        if domain == "saidler_co":
            return {
                "objective": "Spearhead AI adoption at Saidler & Co: Drive 30% efficiency in investments/media.",
                "mock_results": {
                    "engagement_uplift": f"{random.randint(25,50)}% report automation (RAG-powered)",
                    "media_buzz_score": f"9.{random.randint(5,9)}/10 (Blockchain-trend aligned)",
                    "kpi_example": "Governance: Bias reduction via human-in-loop",
                },
            }
        if domain == "telemedicine":
            return {
                "objective": "Boost patient trust & bookings.",
                "mock_results": {
                    "engagement_uplift": f"{random.randint(20,60)}% bookings",
                    "media_buzz_score": "HIPAA-Compliant",
                },
            }
        if domain == "google_pixel":
            return {
                "objective": "Drive Pixel hype & pre-orders.",
                "mock_results": {
                    "engagement_uplift": f"{random.randint(100,500)}K pre-orders",
                    "media_buzz_score": f"{random.randint(8,9)}.{random.randint(0,9)}/10",
                },
            }
        return {
            "objective": "General brand uplift.",
            "mock_results": {
                "engagement_uplift": f"{random.randint(10,40)}% general uplift",
            },
        }

# =============================================================================
# REPLACED & FIXED: RefinerAgent (Full Corrected Version)
# =============================================================================
class RefinerAgent:
    def generate_google_ads_copy(self, dossier: dict, goal: str) -> dict:
        uplift = dossier.get("mock_results", {}).get("engagement_uplift", "30%")
        prompt = (PROMPTS["UNIVERSAL_GOOGLE_ADS"] or "").format(TOPIC=goal, UPLIFT=uplift)
        try:
            _, model = select_model("lite")
            resp = model.generate_content(prompt)
            ads_text = resp.text.strip()
            
            # Split only on first --- separator
            parts = re.split(r'\s*---\s*', ads_text, 1)
            
            ad1_content = parts[0].replace("AD 1:", "").strip() if len(parts) > 0 else "Ad 1 could not be generated."
            ad2_content = parts[1].replace("AD 2:", "").strip() if len(parts) > 1 else "Ad 2 could not be generated."

            return {"ad1": ad1_content, "ad2": ad2_content}
        except Exception as e:
            print(f"Ads error: {e}")
            return {"ad1": "Fallback Ad 1: Error generating ad.", "ad2": "Fallback Ad 2: Error generating ad."}

    def generate_viral_x_post(self, goal: str) -> dict:
        prompt = (PROMPTS["VIRAL_X_POST"] or "").format(TOPIC=goal)
        try:
            _, model = select_model("lite")
            resp = model.generate_content(prompt)
            lines = resp.text.strip().split("\n")
            viral = {}
            for line in lines:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "MEDIA": viral["media"] = value
                elif key == "HOOK POST": viral["post1"] = value
                elif key == "ANALYSIS REPLY": viral["post2"] = value
                elif key == "GROK BOOST": viral["grok_boost"] = value
                elif key == "HASHTAGS": viral["hashtags"] = value
            return viral
        except Exception as e:
            print(f"X Post error: {e}")
            return {"post1": "Fallback Hook", "post2": "Fallback Analysis"}

    def polish(self, dossier: dict, goal: str, domain: str, strategy: dict, plan: dict) -> dict:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            ads_f = ex.submit(self.generate_google_ads_copy, dossier, goal)
            posts_f = ex.submit(self.generate_viral_x_post, goal)
            ads, posts = ads_f.result(), posts_f.result()

        style = {
            "saidler_co": """
                classDef userInput fill:#0a1a2e,stroke:#00bfff,color:#fff;
                classDef agent fill:#1e3a5f,stroke:#00bfff,color:#fff;
                classDef task fill:#00bfff,stroke:#fff,color:#000;
                classDef output fill:#00ff88,stroke:#000,color:#000;
            """,
            "telemedicine": """
                classDef userInput fill:#E8F0FE,stroke:#1967D2,color:#000;
                classDef agent fill:#CEEAD6,stroke:#137333,color:#000;
                classDef task fill:#E1F5FE,stroke:#039BE5,color:#000;
                classDef output fill:#D1F3E1,stroke:#188038,color:#000;
            """,
            "general": """
                classDef userInput fill:#12121f,stroke:#8a2be2,color:#fff;
                classDef agent fill:#8a2be2,stroke:#c71585,color:#fff;
                classDef task fill:#00d4ff,stroke:#8a2be2,color:#000;
                classDef output fill:#007bff,stroke:#00d4ff,color:#fff;
            """,
        }.get(domain, "")

        reasoning = smart_truncate(strategy.get("reasoning", "").replace('"', "'").replace("\n", " ").strip(), 120)
        planner_tool = strategy.get("planner_tool", "GENERAL").replace("_", " ").replace("SAIDLER ", "")

        # Corrected KPI Logic
        kpis = plan.get("mission_workflow", {}).get("kpis", [])
        if domain == "saidler_co" and not kpis:
            kpis = [
                "RAG Efficiency: 40% faster reports",
                "Trend Accuracy: 95% via Grok 4",
                "Compliance Score: 100% ethical prompts",
                "Adoption Rate: 80% staff trained in 30 days",
            ]
        kpi_md = ""
        if kpis:
            kpi_md = "### KPIs\n"
            for k in kpis:
                parts = k.split(":", 1)
                if len(parts) == 2:
                    key_part, value_part = parts[0].strip(), parts[1].strip()
                    kpi_md += f'<div class="kpi-card"><strong>{key_part}:</strong> {value_part}</div>\n'

        insights, next_steps = "", ""
        if domain == "saidler_co":
            uplift = dossier.get("mock_results", {}).get("engagement_uplift", "30%")
            insights = f"""### Key Insights
- **Efficiency Gain**: {uplift} reduction in manual report time via **RAG + MCP orchestration**
- **Accuracy**: **93% hallucination drop** using Grok 4 + internal data fine-tuning
- **Governance**: **Human-in-loop + MCP validation** ensures **FINMA-ready compliance**
"""
        step_map = {
            "rag|pipeline": "### Recommended Next Steps\n1. **Pilot RAG** on 5 pitch decks\n2. **Fine-tune Grok 4** on Q4 data\n3. **Deploy bias dashboard**",
            "workshop": "### Recommended Next Steps\n1. **Book 3-day workshop**\n2. **Assign 10 analysts** to RAG track\n3. **Measure pre/post fluency**",
            "dashboard": "### Recommended Next Steps\n1. **Connect APIs** to arXiv, etc.\n2. **Set alert thresholds**\n3. **Demo to Partners**"
        }
        for pattern, steps in step_map.items():
            if re.search(pattern, goal.lower()):
                next_steps = steps
                break
        
        short_goal = smart_truncate(goal.replace('"', "'"), 80)
        mermaid = f'''
<div class="mermaid">
graph TD
    subgraph "Phase 1: Strategy"
        A["Goal: {short_goal}"]:::userInput --> B["Chief Scout"]:::agent
        B -. "{reasoning}" .-> C["Tool: {planner_tool}"]:::task
    end
    subgraph "Phase 2: Execution"
        C --> D["Planner"]:::agent --> E["Orchestrator"]:::agent --> R["Refiner"]:::agent
        R --> Ads["Google Ads"]:::task & Posts["X Posts"]:::task
    end
    subgraph "Phase 3: Output"
        Ads & Posts --> F["Assets + KPIs"]:::output
    end
    {style}
</div>
'''
        report_md = f"""## MMCOS AI Co-Pilot Report
**Objective:** {goal}
**Domain:** {dossier.get('objective','General')}

{kpi_md}
{insights}
{next_steps}

### AI Workflow
"""
        report_html = markdown.markdown(report_md) + mermaid
        
        return {"report": report_html, "google_ads": ads, "viral_posts": posts}

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    samples = [
        # Core Non-Saidler (kept for breadth)
        {"prompt": "We are a healthcare provider launching a new HIPAA-compliant tele-therapy service for rural communities.", "trends": "mental health awareness, accessible healthcare, AI triage, voice biomarkers"},

        # === SAIDLER & CO – MMCOS ASYNC COGNITIVE ORCHESTRATION (LEVEL 4) ===
        # 1. AI Strategy Briefing (Async Trend Decomposition)
        {"prompt": "As Lead AI Engineer at Saidler & Co, prepare a 5-slide briefing for Partners on MMCOS orchestration for RWA tokenization – highlight async advantages in scaling multimodal LLMs across 200+ deals.", "trends": "Grok 4 beats GPT-5o on chain reasoning, 42% faster pitch deck valuation via parallel agents, Swiss FINMA AI audit draft"},

        # 2. RAG + MCP Blueprint (Non-Blocking Pipelines)
        {"prompt": "Design an async RAG pipeline with MMCOS orchestration and MCP layers to generate investment memos from 200+ pitch decks – leverage event-driven scaling for 87% hallucination reduction.", "trends": "RAG + MCP cuts bias 93%, Llama 3.1 70B fine-tuned for finance, IPFS provenance with concurrent coroutines"},

        # 3. AI Ethics Framework (Fault-Tolerant Governance)
        {"prompt": "Create a 15-page MMCOS Ethics Playbook: async human-in-loop for bias testing, transparency logs, and FINMA checklists – ensure resilient orchestration across portfolio ops.", "trends": "EU AI Act high-risk for finance, Swiss AI registry 2025, async recovery boosts trust by 18% valuation premium"},

        # 4. Trend Scouting Dashboard (Event-Driven Radar)
        {"prompt": "Prototype a live MMCOS AI Radar dashboard: async pulls from arXiv/Hugging Face/GitHub/Davos for finance/blockchain trends – dynamic scaling for 72-hour obsolescence alerts.", "trends": "API-first research, Grok 4 embeddings in parallel, Kubernetes auto-scale for real-time insights"},

        # 5. Fine-Tuning Plan (Distributed Compute)
        {"prompt": "Develop an async fine-tuning plan for Grok 4 on Saidler portfolio data via MMCOS – focus on non-blocking anomaly detection in DeFi with MCP structuring.", "trends": "Few-shot learning, bias checks via event-driven human-in-loop, 30% productivity uplift from orchestration"},

        # 6. Media Asset Personalization (Concurrent Gen)
        {"prompt": "Build a MMCOS-generative system for async personalized video reports on investments for Zug HNW clients – parallel charts/voiceover with blockchain signing.", "trends": "Runway + ElevenLabs multimodal, 94% client preference, fault-tolerant scaling for media assets"},

        # 7. AI Adoption Workshop (Orchestrated Training)
        {"prompt": "Design a 3-day MMCOS workshop for Saidler analysts: Async prompt engineering, RAG setup, Grok 4 fine-tuning – demo event-driven advantages for workflows.", "trends": "80% AI fluency target Q2 2025, zero-to-few-shot gains via non-blocking agents"},

        # 8. Portfolio Maturity Scorecard (Scalable Assessment)
        {"prompt": "Create an async MMCOS scorecard for 50+ portfolio cos: Parallel scoring on RAG adoption, governance, ROI – leverage orchestration for 3.2x faster fundraising insights.", "trends": "McKinsey 2025 AI in VC, top 10% mature startups, dynamic resource allocation"},

        # 9. Token Deal Flow Scoring (Agentic Choreography)
        {"prompt": "Develop MMCOS AI agents for async token deal scoring: NLP on whitepapers, CV on decks, on-chain via Dune – event-driven for non-deterministic flows.", "trends": "CV slide density, sentiment on Discord, The Graph with 99.9% resilience"},

        # 10. Anomaly Detection (Real-Time Async Alerts)
        {"prompt": "Deploy MMCOS-orchestrated Grok 4 anomaly detector for DeFi portfolios – async cross-chain monitoring with auto-alerts, no central bottleneck.", "trends": "LayerZero integration, Grok 4 time-series, 2025 quantum risk via fault-tolerant patterns"},

        # 11. Bias Audit Dashboard (Parallel Validation)
        {"prompt": "Build a live MMCOS bias dashboard: Async flagging of gender/regional biases in reports – MCP validation with concurrent human-in-loop.", "trends": "Swiss AI Act 2025 mandates, orchestration for 93% bias drop"},

        # 12. Custom GPT Monitoring (Predictive Async)
        {"prompt": "Develop async custom GPT via MMCOS for real-time blockchain portfolio alerts – predictive shifts with event-driven scaling.", "trends": "Finance GPTs, blockchain monitoring, 2025 predictive AI"},

        # 13. Due Diligence Workflow (Automation Orchestration)
        {"prompt": "Design MMCOS intelligent async workflow for media tech due diligence: Parallel scraping, summarization, risk scoring.", "trends": "Web ethics, AI risk models, agentic non-blocking flows"},

        # 14. Multimodal Recommendations (Concurrent Processing)
        {"prompt": "Build MMCOS multi-modal rec system for tokens: Async NLP sentiment + CV decks – orchestration for 38% valuation accuracy.", "trends": "2025 multi-modal models, document CV"},

        # 15. Portfolio Training Program (Scalable Rollout)
        {"prompt": "Outline MMCOS AI training for blockchain/media portfolio cos – async workshops on gen AI, customization with orchestration demos.", "trends": "Startup AI training, 2025 blockchain integration"},

        # 16. Media Newsletters (Personalized Async Delivery)
        {"prompt": "Automate MMCOS content for media assets: Async personalized newsletters for Zug HNW – parallel gen with provenance.", "trends": "Agentic AI, multimodal content, blockchain"},

        # 17. Governance Guidelines (Resilient Enforcement)
        {"prompt": "Develop MMCOS guidelines for ethical AI: Async bias/transparency enforcement across Saidler ops.", "trends": "Finance ethics, 2025 Swiss regs"},

        # 18. Efficiency Opportunities (Impact Identification)
        {"prompt": "Identify MMCOS async ops for accounting/reports/market research – establish KPIs for 40% gains.", "trends": "AI market research, automation"},

        # 19. Model Optimization (Continuous Async Eval)
        {"prompt": "Optimize MMCOS proprietary models: Async evaluation, dataset curation for domain LLMs.", "trends": "Performance tuning, RAG-specific"},

        # 20. Tools Integration (Organization-Wide Async)
        {"prompt": "Lead MMCOS integration of AI tools: Async docs/training for teams/portfolio – scalable adoption.", "trends": "Internal adoption, staff fluency"},

        # 21. End-to-End Projects (Orchestrated Deployment)
        {"prompt": "Lead MMCOS end-to-end AI project: Async from reqs to deploy – internal improvements with iteration.", "trends": "Project leadership, 2025 scaling"},
    ]

    if request.method == "POST":
        goal = request.form.get("goal_and_context")
        trends = request.form.get("user_trends", "General")
        if not goal:
            return render_template("index.html", error="Goal is required.", samples=samples)

        try:
            get_client()
            scout = ChiefScoutAgent()
            planner = PlannerAgent()
            orch = OrchestratorAgent()
            refiner = RefinerAgent()

            strategy = scout.strategize(goal)
            plan = planner.plan(goal, trends, strategy["planner_tool"])
            dossier = orch.execute(plan, goal, strategy["domain_key"])
            output = refiner.polish(dossier, goal, strategy["domain_key"], strategy, plan)

            return render_template(
                "index.html",
                goal_and_context=goal,
                user_trends=trends,
                report=output["report"],
                google_ads=output["google_ads"],
                viral_posts=output["viral_posts"],
                samples=samples,
            )
        except Exception as e:
            print(f"Route error: {e}")
            return render_template("index.html", error=f"Error: {e}", samples=samples)

    return render_template("index.html", samples=samples)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

