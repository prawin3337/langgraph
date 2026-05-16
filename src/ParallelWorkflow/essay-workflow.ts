import {START, END, StateSchema, StateGraph, ReducedValue} from '@langchain/langgraph';
import z from 'zod';
import {ChatOpenAI} from '@langchain/openai';

const run = async () => {
    const model = new ChatOpenAI('gpt-4o-mini');

    const EvaluationSchema = z.object({
        feedback: z.string('Detailed feedbackfor the essay'),
        score: z.number('Score out of 10').min(5).max(10)
    });

    const structuredModel = model.withStructuredOutput(EvaluationSchema);

    // Previous implementation for reference:
    /*
    const stateSchema = new StateSchema({
        essay: z.string(),
        language_feedback: z.string(),
        analysis_feedback: z.string(),
        clarity_feedback: z.string(),
        overall_feedback: z.string(),
        individual_scores: new ReducedValue(
            z.array(z.number()).default(() => []),
            {
                inputSchema: z.array(z.number()),
                reducer: (current, next) => current.concat(next)
            }
        ),
        avg_score: z.number()
    });
    */

    // New implementation: store individual_scores as array of objects
    const scoreObjectSchema = z.object({
        langFeedbackScore: z.number().optional(),
        analysisScore: z.number().optional(),
        clarityScore: z.number().optional()
    });
    const stateSchema = new StateSchema({
        essay: z.string(),
        language_feedback: z.string(),
        analysis_feedback: z.string(),
        clarity_feedback: z.string(),
        overall_feedback: z.string(),
        individual_scores: new ReducedValue(
            z.array(scoreObjectSchema).default(() => []),
            {
                inputSchema: z.array(scoreObjectSchema),
                reducer: (current, next) => current.concat(next)
            }
        ),
        avg_score: z.number()
    });

    type EssayState = typeof stateSchema.State;

    const evaluateLangualge = async (state: EssayState) => {
        const prompt = `Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n ${state.essay}`;
        const output = await structuredModel.invoke(prompt);
        return {
            language_feedback: output.feedback,
            individual_scores: [{ langFeedbackScore: output.score }]
        };
    }

    const evaluateAnalysis = async (state: EssayState) => {
        const prompt = `Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n ${state.essay}`;
        const output = await structuredModel.invoke(prompt);
        return {
            analysis_feedback: output.feedback,
            individual_scores: [{ analysisScore: output.score }] // Previous implementation for reference: {analysis_feedback: output.feedback, individual_scores: [output.score]}
        };
    }

    const evaluateThought = async (state: EssayState) => {
        const prompt = `Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n ${state.essay}`;
        const output = await structuredModel.invoke(prompt);
        return {
            clarity_feedback: output.feedback,
            individual_scores: [{ clarityScore: output.score }]
        };
    }

    // Previous implementation for reference:
    /*
    const finalEvaluation = async (state: EssayState) => {
        const prompt = `Based on the following feedbacks create a summarized feedback \n language feedback - ${state.language_feedback} \n depth of analysis feedback - ${state.analysis_feedback} \n clarity of thought feedback - ${state.clarity_feedback}`;
        const overallFeedback = (await model.invoke(prompt)).content;
        const sum = state.individual_scores.reduce((a, b) => a + b, 0);
        const avgScore = sum/state.individual_scores.length;
        return {overall_feedback: overallFeedback, avg_score: avgScore};
    }
    */
    // New implementation:
    const finalEvaluation = async (state: EssayState) => {
        const prompt = `Based on the following feedbacks create a summarized feedback \n language feedback - ${state.language_feedback} \n depth of analysis feedback - ${state.analysis_feedback} \n clarity of thought feedback - ${state.clarity_feedback}`;
        const overallFeedback = (await model.invoke(prompt)).content;
        // Flatten all scores into a single array for average calculation
        const allScores = state.individual_scores.flatMap(obj => Object.values(obj).filter(v => typeof v === 'number'));
        const sum = allScores.reduce((a, b) => a + b, 0);
        const avgScore = allScores.length > 0 ? sum / allScores.length : 0;
        return {overall_feedback: overallFeedback, avg_score: avgScore};
    }

    const graph = new StateGraph(stateSchema);
    graph.addNode('evaluate_language', evaluateLangualge)
        .addNode('evaluate_analysis', evaluateAnalysis)
        .addNode('evaluate_thought', evaluateThought)
        .addNode('final_evaluation', finalEvaluation)

        .addEdge(START, 'evaluate_language')
        .addEdge(START, 'evaluate_analysis')
        .addEdge(START, 'evaluate_thought')

        .addEdge('evaluate_language', 'final_evaluation')
        .addEdge('evaluate_analysis', 'final_evaluation')
        .addEdge('evaluate_thought', 'final_evaluation')

        .addEdge('final_evaluation', END);

    const workflow = graph.compile();

    const essay = `India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.
India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.
One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.
However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.
Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.
To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.
India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.
In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society.`;
 

    const result = await workflow.invoke({essay, individual_scores: []});

    console.log('final state ===>', result);

}
export default run;