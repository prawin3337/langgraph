import {
    addMessages, 
    Annotation, 
    END, 
    START, 
    StateGraph, 
    MemorySaver
} from '@langchain/langgraph';
import { tool } from 'langchain';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt';
import {ChatOpenAI} from '@langchain/openai'
import { BaseMessage, HumanMessage } from 'langchain';
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

    const calculatorTool = tool(
        async ({first_num, second_num, operation}) => {
            console.log('tool call')
            try {
                if(operation === 'add') {
                    return first_num + second_num
                }
                else if(operation === 'sub') {
                    return first_num - second_num
                }
                else if(operation === 'mul') {
                    return first_num * second_num;
                }
                else if(operation === 'div') {
                    if(second_num === 0) return {error: "Division by zero not allowed"}
                    return first_num / second_num;
                }
            } catch(err) {
                console.log(err)
            }
        },
        {
            name: 'calculator',
            description: 'Perform a basic arithmetic operation on two numbers. Supported operations: add, sub, mul, div',
            schema: z.object({
                first_num: z.number(),
                second_num: z.number(),
                operation: z.string()
            })
        }
    );

    const getStockPrise = tool(
        async ({symbol}: {symbol: string}) => {
            const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${process.env.ALPHAVANTAGE}`
            const res = await fetch(url);
            return res.json();
        },
        {
            name: 'get-stock-prise',
            description: 'Fetch latest stock price for a given symbol (e.g. \'AAPL\', \'TSLA\') using Alpha Vantage with API key in the URL.',
            schema: z.object({
                symbol: z.string()
            })
        }
    )

    const mcpClient = new MultiServerMCPClient({
        mcpServers: {
            remote: {
                url: 'https://splendid-gold-dingo.fastmcp.app/mcp',
                transport: 'http'
            }
        }
    });

    const mcpTools = await mcpClient.getTools();
    // console.log('mcp tools', mcpTools)

    const tools = [calculatorTool, getStockPrise, ...mcpTools];
    const toolNode = new ToolNode(tools)
    const llmWithTools = llm.bindTools(tools);

    const ChatState = Annotation.Root({
        messages: Annotation<BaseMessage[]>({
            reducer: addMessages,
            default: () => []
        })
    });

    const chatNode = async (state: typeof ChatState.State) => {
        const response = await llmWithTools.invoke(state.messages);
        return {messages: [response]};
    }

    const graph = new StateGraph(ChatState);
    graph.addNode('chat_node', chatNode)
        .addNode('tools', toolNode)
        .addEdge(START, 'chat_node')
        .addConditionalEdges('chat_node', toolsCondition)
        .addEdge('tools', 'chat_node')
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
    
}