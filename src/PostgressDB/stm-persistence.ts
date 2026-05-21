/**
 * https://chatgpt.com/share/6a0ea7d3-7514-8320-9bc7-192cc82e9d0b
 */

import { StateGraph, START, MessagesZodState, MessagesAnnotation, END } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "langchain";
import { Pool } from 'pg';

export default async () => {
    const llm = new ChatOpenAI('gpt-4o-mini');

    const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres";
    const pool = new Pool({
        connectionString: DB_URI
    });

    const checkpointer = new PostgresSaver(pool);
    await checkpointer.setup();

    const callModel = async (state: typeof MessagesAnnotation.State) => {
        const response = await llm.invoke(state.messages);
        return {messages: [response]}
    }

    const graph = new StateGraph(MessagesAnnotation);
    graph.addNode('call_model', callModel)
        .addEdge(START, 'call_model')
        .addEdge('call_model', END);

    
    const workflow = graph.compile({checkpointer});

    const t1 = {configurable: {'thread_id': 't1'}};
    // await workflow.invoke({messages: [new HumanMessage({content: 'Hi, my name is Pravin'})]}, t1);
    const out1 = await workflow.invoke({messages: [new HumanMessage({content: 'What is my name?'})]}, t1);
    console.log(out1.messages);
   
    // Fetch messages from DB
    // const snap = await workflow.getState(t1);
    // const dbMessages = await snap.values;
    // console.log('obj= ', dbMessages);
}