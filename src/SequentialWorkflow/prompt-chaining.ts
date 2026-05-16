import { END, START, StateGraph, StateSchema } from '@langchain/langgraph';
import {ChatOpenAI} from '@langchain/openai';
import z from 'zod';

const run = async () => {

    const model = new ChatOpenAI('gpt-4');

    type BlogState = { title: string; outline: string, content: string };
    const schema = {
        title: z.string(),
        outline: z.string(),
        content: z.string()
    }
    const LLMState = new StateSchema(schema);

    const createOutline = async (state: BlogState) => {
        const title = state.title;
        const prompt = `Generate a detailed outline for a blog on the topic - ${title}`;
        const outline = (await model.invoke(prompt)).content;
        state.outline = typeof outline === 'string' ? outline : JSON.stringify(outline);
        return state;
    }

    const createBlog = async (state: BlogState) => {
        const title = state.title;
        const outline = state.outline;
        const prompt = `Write a detailed blog on the title - ${title} using the follwing outline \n ${outline}`;
        const blog = (await model.invoke(prompt)).content;
        state.content = typeof blog === 'string' ? blog : JSON.stringify(blog);
        return state;
    }

    const graph = new StateGraph(LLMState)
        .addNode('create_outline', createOutline)
        .addNode('create_blog', createBlog)
        .addEdge(START, 'create_outline')
        .addEdge('create_outline', 'create_blog')
        .addEdge('create_blog', END);
    
    const workflow = graph.compile();

    const finalState = await workflow.invoke({title: 'Rise of AI in India.'});
    console.log(finalState);

};

export default run;