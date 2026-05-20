import {
    END, 
    START, 
    StateGraph
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import z from 'zod';


export default async () => {
    const llm = new ChatOpenAI('gpt-4o-mini');

    const ParentState = z.object({
        question: z.string(),
        answerEng: z.string(),
        answerHindi: z.string()
    });
    
    const translateText = async (state: z.infer<typeof ParentState>) => {
        const prompt = `Translate the following text to hindi. Keep it natural and clear, Do not add extra content.

        Text: ${state.answerEng}`;

        const translatedText = (await llm.invoke(prompt)).content;
        return {answerHindi: translatedText};
    }

    const subGraph = new StateGraph(ParentState);
    subGraph.addNode('translate_text', translateText)
        .addEdge(START, 'translate_text')
        .addEdge('translate_text', END);
    
    const subGraphWorkflow = subGraph.compile();

    const generateAnswer = async (state: z.infer<typeof ParentState>) => {
        const answer = await llm.invoke(`You are a helpful assistant. Answer clearly.\n\nQuestion: ${state.question}`);
        return {'answerEng': answer.content}
    }

    const parentGraph = new StateGraph(ParentState);
    parentGraph.addNode('answer', generateAnswer)
        .addNode('translate', subGraphWorkflow)
        .addEdge(START, 'answer')
        .addEdge('answer', 'translate')
        .addEdge('translate', END);

    const parentWorkflow = parentGraph.compile();
    const result = await parentWorkflow.invoke({question: 'What is quantum physics?'});
    console.log(result);
}