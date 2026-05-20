import {
    addMessages, 
    Annotation, 
    END, 
    START, 
    StateGraph, 
    MemorySaver,
    interrupt,
    Command,
    INTERRUPT,
    isInterrupted
} from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { AIMessage, BaseMessage, HumanMessage } from 'langchain';
import { stdin, stdout } from 'process';
import { createInterface } from 'readline/promises';
import z from 'zod';

export default async () => {
    const llm = new ChatOpenAI('gpt-4o-mini');

    const rl = createInterface({
            input: stdin,
            output: stdout,
            terminal: false
        });

    const ChatState = Annotation.Root({
        messages: Annotation<BaseMessage[]>({
            reducer: addMessages,
            default: () => []
        })
    });

    const chatNode = async (state: typeof ChatState.State) => {
        const decision = interrupt({
            type: 'approval',
            reason: 'Model is about to answer a user question.',
            question: state.messages.at(-1)?.content,
            instruction: 'Approve this question? yes/no'
        });

        // console.log('decision.approved =====>', decision.approved )

        if(decision.approved === 'no') {
            return {'messages': [new AIMessage({content: 'Not approved'})]}
        } else {
            const response = await llm.invoke(state.messages);
            return {messages: [response]}
        }
    }

    const graph = new StateGraph(ChatState);
    graph.addNode('chat_node', chatNode)
        .addEdge(START, 'chat_node')
        .addEdge('chat_node', END);

    const checkpointer = new MemorySaver();
    
    const workflow = graph.compile({checkpointer});

    const config = {configurable: {thread_id: 1}};

    const initialState = {
        messages: [new HumanMessage({content: 'Explain gradient descent in very simple terms.'})]
    }
    const result = await workflow.invoke(initialState, config);
    console.log(result);

    let userMessage;
    if(isInterrupted(result)) {
        const message:any = result[INTERRUPT].at(0)?.value;
        console.log(message)
        userMessage = await rl.question(`Backend message - ${message.question} \n Approve this question? (yes/no): `);
    }

    const finalResult = await workflow.invoke(new Command({ resume: {approved: userMessage} }), config);
    console.log(finalResult.messages.at(-1)?.content);
}