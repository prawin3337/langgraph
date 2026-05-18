import {addMessages, Annotation, END, START, StateGraph, MemorySaver} from '@langchain/langgraph';
import {ChatOpenAI} from '@langchain/openai'
import { BaseMessage, HumanMessage } from 'langchain';
import { stdin, stdout } from 'process';
import { createInterface } from 'readline/promises';


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
        const response = await llm.invoke(state.messages);
        return {messages: [response]};
    }

    const graph = new StateGraph(ChatState);
    graph.addNode('chat_node', chatNode)
        .addEdge(START, 'chat_node')
        .addEdge('chat_node', END)

    const checkPointer = new MemorySaver();

    const workflow = graph.compile({checkpointer: checkPointer});

    const threadId = '1';
    const config = {configurable: {thread_id: threadId}}

    while(true) {
        const userMessage = await rl.question('Enter query: ');

        if(userMessage === 'exit') {
            rl.close();
            break;
        }

        const humanMessage = {
            messages: [new HumanMessage({
                content: userMessage
            })]
        };

        const response = await workflow.invoke(humanMessage, config);
        console.log(response.messages.at(-1)?.content);
    }

    const history = workflow.getStateHistory(config);

    // Since workflow.getStateHistory(config) returns an async iterable, you cannot directly use .map() or normal array indexing during iteration.
    let index = 0;
    for await (const state of history) {
        console.log("Checkpoint index:", index);
        console.log("Values:", state.values);
        // console.log("Next:", state.next);
        // console.log("Metadata:", state.metadata);
        // console.log("Tasks:", state.tasks);
        // console.log("Config:", state.config);
        index++;
    }
    
}