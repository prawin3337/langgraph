import { StateSchema, GraphNode, StateGraph, START, END } from "@langchain/langgraph";
import z from 'zod';

const run = async () => {
    const BMIState = new StateSchema({
        weightKg: z.number(),
        heightM: z.number(),
        bmi: z.number(),
        category: z.string()
    });

    const calculateBMI = (state: typeof BMIState.State): typeof BMIState.State => {
        const weightKg = state.weightKg;
        const heightM = state.heightM;
        state.bmi = Math.round(weightKg/(heightM**2));
        return state;
    }

    const labelBMI = (state: typeof BMIState.State): typeof BMIState.State => {
        const bmi = state.bmi;

        if(bmi < 18.5) {
            state.category = 'underWaight'
        } else
        if(bmi < 25) {
            state.category = 'normal'
        } else
        if(bmi < 30) {
            state.category = 'overWeight'
        }
        else {
            state.category = 'obese'
        }

        return state;
    }

    const workflow = new StateGraph(BMIState)
        .addNode('calculate_bmi', calculateBMI)
        .addNode('label_bmi', labelBMI)
        .addEdge(START, 'calculate_bmi')
        .addEdge('calculate_bmi', 'label_bmi')
        .addEdge('label_bmi', END)
        .compile();

    const result = await workflow.invoke({heightM: 1.73, weightKg: 60});
    console.log(result);
}

export default run;