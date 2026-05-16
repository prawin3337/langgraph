import { END, START, StateGraph, StateSchema } from '@langchain/langgraph';
import {ChatOpenAI} from '@langchain/openai';
import z from 'zod';

const run = async () => {

    const model = new ChatOpenAI('gpt-4o-mini');

    type QAState = { question: string; answer: string };
    const schema = {
        question: z.string(),
        answer: z.string()
    }
    const LLMState = new StateSchema(schema);

    const llmQA = async (state: QAState) => {
        const question = state.question;
        const prompt = `Answer the following question:\n${question}`;
        const content = (await model.invoke(prompt)).content;
        const answer = typeof content === 'string' ? content : JSON.stringify(content);
        state.answer = answer;
        return state;
    }

    const graph = new StateGraph(LLMState)
        .addNode('llm_qa', llmQA)
        .addEdge(START, 'llm_qa')
        .addEdge('llm_qa', END);
    
    const workflow = graph.compile();

    const finalState = await workflow.invoke({question: 'How far is moon from the earth?'});
    console.log(finalState);

};

export default run;