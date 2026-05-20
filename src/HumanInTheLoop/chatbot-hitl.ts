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
import { ToolNode, toolsCondition } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { AIMessage, BaseMessage, HumanMessage, tool } from 'langchain';
import { stdin, stdout } from 'process';
import { createInterface } from 'readline/promises';
import z, { symbol } from 'zod';

export default async () => {
    const llm = new ChatOpenAI('gpt-4o-mini');

    const rl = createInterface({
            input: stdin,
            output: stdout,
            terminal: false
        });
    
    const getStockPrise = tool(
        async ({symbol}: {symbol: string}) => {
            const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${process.env.ALPHAVANTAGE}`
            const res = await fetch(url);
            // console.log('res =========>', res.json())
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

    const purchaseStock = tool(
        async ({symbol, quantity}) => {
            const decision = interrupt(`Approve buying ${quantity} shares of ${symbol}? (yes/no)`);

            if(decision === 'yes') {
                return {
                    status: 'success',
                    message: `Purchase order placed for ${quantity} shares of ${symbol}.`,
                    symbol,
                    quantity
                }
            }

            return {
                status: 'cancelled',
                message: `Purchase of ${quantity} shares of ${symbol} was declined by user.`,
                symbol,
                quantity
            }
        },
        {
            name: 'purchase-stock',
            description: `
            Simulate purchasing a given quantity of a stock symbol.

            HUMAN-IN-THE-LOOP:
            Before confirming the purchase, this tool will interrupt
            and wait for a human decision ("yes" / anything else).
            `,
            schema: z.object({
                symbol: z.string(),
                quantity: z.number()
            })
        }
    )

    const tools = [getStockPrise, purchaseStock];

    const toolNode = new ToolNode(tools);

    const llmWithTools = llm.bindTools(tools);

    const ChatState = Annotation.Root({
        messages: Annotation<BaseMessage[]>({
            reducer: addMessages,
            default: () => []
        })
    });

    const chatNode = {
        name: 'chat_node',
        description: 'LLM node that may answer or request a tool call.',
        handler: async (state: typeof ChatState.State) => {
            const messages = state.messages;
            const response = await llmWithTools.invoke(messages);
            return {messages: [response]};
        }
    }

    const memory = new MemorySaver();

    const graph = new StateGraph(ChatState);
    graph.addNode('chat_node', chatNode.handler)
        .addNode('tools', toolNode)
        
        .addEdge(START, 'chat_node')

        .addConditionalEdges('chat_node', toolsCondition)
        .addEdge('tools', 'chat_node')

    const workflow = graph.compile({checkpointer: memory});

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

        let result = await workflow.invoke(humanMessage, config);
        
        if(isInterrupted(result)) {
            const interrupts = result[INTERRUPT];
            const promptToHuman = interrupts[0]?.value;
            console.log(`HITL: ${promptToHuman}`)
            const decision = await rl.question('Your decision: ');

            result = await workflow.invoke(new Command({resume: decision}), config);
        }

        console.log(result?.messages.at(-1)?.content);
    }


}