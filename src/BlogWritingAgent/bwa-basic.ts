import { Annotation, StateGraph, START, END, Send } from "@langchain/langgraph";
import { SystemMessage, HumanMessage } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import z from 'zod';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

export default async () => {
    const Task = z.object({
        id: z.number(),
        title: z.string(),
        goal: z.string().describe('One sentence describing what the reader should be able to do/understand after this section.'),
        bullets: z.array(z.string())
                    .describe('3–5 concrete, non-overlapping subpoints to cover in this section.')
                    .min(3).max(5),
        targetWords: z.number().describe('Target word count for this section (120–450).'),
        sectionType: z.literal(["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"])
                    .describe("Use 'common_mistakes' exactly once in the plan.")
        
    });

    const Plan = z.object({
        blogTitle: z.string(),
        tasks: z.array(Task),
        audience: z.string().describe('Who this blog is for.'),
        tone: z.string().describe('Writing tone (e.g., practical, crisp).')
    });

    type PlanState = z.infer<typeof Plan>;

    const State = Annotation.Root({
        topic: Annotation<string>,
        plan: Annotation<PlanState>,
        sections: Annotation<string[]>({
            reducer: (cur, next) => {
                return cur.concat(next)
            },
            default: () => []
        }),
        final: Annotation<string>
    });

    type BlogState = typeof State.State;

    const llm = new ChatOpenAI('gpt-4o-mini');

    const orchestrator = async (state: BlogState) => {
        const plan = await llm.withStructuredOutput(Plan).invoke(
            [
                new SystemMessage({ content: `You are a senior technical writer and developer advocate. Your job is to produce a 
                    highly actionable outline for a technical blog post.
                    Hard requirements:
                    - Create 5–7 sections (tasks) that fit a technical blog.
                    - Each section must include:
                      1) goal (1 sentence: what the reader can do/understand after the section)
                      2) 3–5 bullets that are concrete, specific, and non-overlapping
                      3) target word count (120–450)
                    - Include EXACTLY ONE section with section_type='common_mistakes'.
                    Make it technical (not generic):
                    - Assume the reader is a developer; use correct terminology.
                    - Prefer design/engineering structure: problem → intuition → approach → implementation → 
                    trade-offs → testing/observability → conclusion.
                    - Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', 
                    'Explain why Y fails under Z condition', 'Add a checklist for production readiness').
                    - Explicitly include at least ONE of the following somewhere in the plan (as bullets):
                      * a minimal working example (MWE) or code sketch
                      * edge cases / failure modes
                      * performance/cost considerations
                      * security/privacy considerations (if relevant)
                      * debugging tips / observability (logs, metrics, traces)
                    - Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what 
                    to build/compare/measure/verify.
                    Ordering guidance:
                    - Start with a crisp intro and problem framing.
                    - Build core concepts before advanced details.
                    - Include one section for common mistakes and how to avoid them.
                    - End with a practical summary/checklist and next steps.
                    Output must strictly match the Plan schema."` }),
                new HumanMessage({ content: `Topic: ${state.topic}` })
            ]
        )
        return { plan };
    }

    const fanout = (state: BlogState) => {
        return state.plan.tasks.map((task) =>
            new Send("blog_worker", {
                task,
                topic: state.topic,
                plan: state.plan,
            })
        );
    }

    const blogWorker = async (payload: any) => {
        const { task, topic, plan } = payload;
        const blogTitle = plan.blogTitle;
        const sectionMd = await llm.invoke([
            new SystemMessage({content: `You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n
        Hard constraints:
        - Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
        - Stay close to the Target words (±15%).
        - Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
        Technical quality bar:
        - Be precise and implementation-oriented (developers should be able to apply it).
        - Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.
        - When relevant, include at least one of:
          * a small code snippet (minimal, correct, and idiomatic)
          * a tiny example input/output
          * a checklist of steps
          * a diagram described in text (e.g., 'Flow: A -> B -> C')
        - Explain trade-offs briefly (performance, cost, complexity, reliability).
        - Call out edge cases / failure modes and what to do about them.
        - If you mention a best practice, add the 'why' in one sentence.
        Markdown style:
        - Start with a '## <Section Title>' heading.
        - Use short paragraphs, bullet lists where helpful, and code fences for code.
        - Avoid fluff. Avoid marketing language.
        - If you include code, keep it focused on the bullet being addressed.`}),
            new HumanMessage({content: `
                Blog: ${blogTitle}
                Audience: ${plan.audience}
                Tone: ${plan.tone}
                Topic: ${topic}
                Section: ${task.title}
                Section type: ${task.sectionType}
                Goal: ${task.goal}
                Target words: ${task.targetWords}
                Bullets: ${task.bullets.join("\n- " + "\n- ")}
                Return only the section content in Markdown.
            `})
        ]);
        return {'sections': [sectionMd.content]}
    }

    const reducer = async (state: BlogState) => {
        const title = state.plan.blogTitle;
        const body = state.sections.join("\n\n");

        const finalMd = `# ${title} \n\n ${body} \n`;

        const fileName = `${title.toLowerCase().replace(/\s+/g, "_")}.md`;
        const currentDir = path.dirname(fileURLToPath(import.meta.url));
        const outputPath = path.join(currentDir, fileName);
        fs.writeFileSync(outputPath, finalMd, 'utf-8');

        return {final: finalMd};
    }

    const graph = new StateGraph(State);
    graph.addNode('orchestrator', orchestrator)
        .addNode('blog_worker', blogWorker)
        .addNode('reducer', reducer)
        .addEdge(START, 'orchestrator')
        .addConditionalEdges('orchestrator', fanout, ['blog_worker'])
        .addEdge('blog_worker', 'reducer')
        .addEdge('reducer', END);

    const workflow = graph.compile();
    const result = await workflow.invoke({topic: "Write a blog on Self Attention"});
    console.log(result);
}