import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import { z } from "zod/v4";

export default async () => {
    const State = new StateSchema({
        topic: z.string(),
        joke: z.string(),
    });

    const graph = new StateGraph(State)
        .addNode("refineTopic", (state) => {
            return { topic: state.topic + " and cats" };
        })
        .addNode("generateJoke", (state) => {
            return { joke: `This is a joke about ${state.topic}` };
        })
        .addEdge(START, "refineTopic")
        .addEdge("refineTopic", "generateJoke")
        .addEdge("generateJoke", END)
        .compile();

    for await (const chunk of await graph.stream(
        { topic: "ice cream" },
        { streamMode: "updates" }
    )) {
        for (const [nodeName, state] of Object.entries(chunk)) {
            console.log(`Node ${nodeName} updated:`, state);
        }
    }
}