import { END, START, StateGraph, StateSchema } from '@langchain/langgraph';
import z from 'zod';

const run = async () => {
    const fields = {
        runs: z.number(),
        balls: z.number(),
        fours: z.number(),
        sixes: z.number(),
        sr: z.number(),
        bpb: z.number(),
        boundaryPercent: z.number(),
        _summary: z.string()
    };
    const schema = z.object(fields)

    type BatsmanState = z.infer<typeof schema>;

    const calculateSR = (state: BatsmanState) => {
        const sr = (state.runs/state.balls)*100;
        return {sr};
    }

    const calculatBPB = (state: BatsmanState) => {
        const bpb = state.balls/(state.fours + state.sixes);
        return {bpb}
    }

    const calculateBoundryPercent = (state: BatsmanState) => {
        const boundaryPercent = (((state.fours * 4) + (state.sixes * 6))/state.runs)*100
        return {boundaryPercent}
    }

    const summary = (state: BatsmanState) => {
        const _summary = `
            Strike Rate - ${state.sr}
            Balls per boundary - ${state.bpb}
            Boundary percent - ${state.boundaryPercent}`
        return {_summary}
    }

    const BatsmanState = new StateSchema(fields);
    const graph = new StateGraph(BatsmanState)
        .addNode('calculate_sr', calculateSR)
        .addNode('calculate_bpb', calculatBPB)
        .addNode('calculate_boundary_percent', calculateBoundryPercent)
        .addNode('summary', summary)

        // Parallel Execution
        .addEdge(START, 'calculate_sr')
        .addEdge(START, 'calculate_bpb')
        .addEdge(START, 'calculate_boundary_percent')

        // Sequential Execution
        .addEdge('calculate_sr', 'summary')
        .addEdge('calculate_bpb', 'summary')
        .addEdge('calculate_boundary_percent', 'summary')
        .addEdge('summary', END);

    const workflow = graph.compile();
    const finalState = await workflow.invoke({'runs': 100, 'balls': 50, 'fours': 6, 'sixes': 4});

    console.log(finalState);
}

export default run;