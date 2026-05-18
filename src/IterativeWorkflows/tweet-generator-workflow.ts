import {END, ReducedValue, START, StateGraph, StateSchema} from '@langchain/langgraph';
import {ChatOpenAI} from '@langchain/openai'
import { HumanMessage, SystemMessage } from 'langchain';
import z from 'zod';

export default async () => {
    const llm = new ChatOpenAI('gpt-4o-mini');

    const tweerEvaluationObj = {
        evaluation: z.enum(['approved', 'needs_improvement']).describe('Final evaluation result.'),
        feedback: z.string().describe('feedback for the tweet')
    }
    const TweerEvaluation = z.object(tweerEvaluationObj);
    const structuredEvaluatorLLM = llm.withStructuredOutput(TweerEvaluation);

    const tweetStateObj = {
        topic: z.string(),
        tweet: z.string().default(''),
        evaluation: z.enum(['approved', 'needs_improvement']).default('needs_improvement'),
        feedback: z.string().default(''),
        iteration: z.number().default(0),
        max_iteration: z.number().default(3),
        tweet_history: new ReducedValue(
            z.array(z.string()).default(() => []),
            {
                inputSchema: z.array(z.string()),
                reducer: (cur, next) => cur.concat(next)
            }
        ),
        feedback_history: new ReducedValue(
            z.array(z.string()).default(() => []),
            {
                inputSchema: z.array(z.string()),
                reducer: (cur, next) => cur.concat(next)
            }
        )
    }
    // z.object() is for TypeScript type inference only — ReducedValue instances must NOT be passed to z.object()
    const TweetSchema = z.object({
        topic: z.string(),
        tweet: z.string().default(''),
        evaluation: z.enum(['approved', 'needs_improvement']).default('needs_improvement'),
        feedback: z.string().default(''),
        iteration: z.number().default(0),
        max_iteration: z.number().default(3),
        tweet_history: z.array(z.string()),
        feedback_history: z.array(z.string()),
    });
    const tweetState = new StateSchema(tweetStateObj);

    const generateTweet = async (state: z.infer<typeof TweetSchema>) => {
        const prmptMessages = [
            new SystemMessage({
                content: 'You are a funny and claver Twitter influencer.'
            }),
            new HumanMessage({
                content: `Write a short, original, and hilarious tweet on the topic: "${state.topic}".
                            Rules:
                            - Do NOT use question-answer format.
                            - Max 280 characters.
                            - Use observational humor, irony, sarcasm, or cultural references.
                            - Think in meme logic, punchlines, or relatable takes.
                            - Use simple, day to day english`
            })
        ];
        const response = (await llm.invoke(prmptMessages)).content;
        return {tweet: response, 'tweet_history': [response]}
    }

    const evaluateTweet = async (state: z.infer<typeof TweetSchema>) => {
        const promptMessages = [
            new SystemMessage({
                content: `You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format.`
            }),
            new HumanMessage({
                content: `Evaluate the following tweet:

                Tweet: "${state.tweet}"

                Use the criteria below to evaluate the tweet:

                1. Originality – Is this fresh, or have you seen it a hundred times before?  
                2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
                3. Punchiness – Is it short, sharp, and scroll-stopping?  
                4. Virality Potential – Would people retweet or share it?  
                5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

                Auto-reject if:
                - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
                - It exceeds 280 characters
                - It reads like a traditional setup-punchline joke
                - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

                ### Respond ONLY in structured format:
                - evaluation: "approved" or "needs_improvement"  
                - feedback: One paragraph explaining the strengths and weaknesses `
            })
        ];
        const response = (await structuredEvaluatorLLM.invoke(promptMessages));
        return {evaluation: response.evaluation, feedback: response.feedback, feedback_history: [response.feedback]}
    }

    const optimizeTweet = async (state: z.infer<typeof TweetSchema>) => {
        const promptMessages = [
            new SystemMessage({
                content: `You punch up tweets for virality and humor based on given feedback.`
            }),
            new HumanMessage({
                content: `Improve the tweet based on this feedback:
                "${state.feedback}"

                Topic: "${state.topic}"
                Original Tweet: ${state.tweet}

                Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.`
            })
        ];
        const response = (await llm.invoke(promptMessages)).content;
        const iteration = state.iteration + 1;
        return {'tweet': response, iteration, 'tweet_history': [response]}
    }

    const routeEvaluation = async (state: z.infer<typeof TweetSchema>) => {
        if(state.evaluation === 'approved' || state.iteration >= state.max_iteration) {
            return 'approved'
        }
        return 'needs_improvement'
    }

    const graph = new StateGraph(tweetState);
    graph.addNode('generate', generateTweet)
        .addNode('evaluate', evaluateTweet)
        .addNode('optimize', optimizeTweet)

        .addEdge(START, 'generate')
        .addEdge('generate', 'evaluate')

        .addConditionalEdges('evaluate', routeEvaluation, {'approved': END, 'needs_improvement': 'optimize'})
        .addEdge('optimize', 'evaluate')
    
    const workflow = graph.compile();
    const result = await workflow.invoke({topic: 'srhberhb'});
    console.log(result);
}