import {END, START, StateGraph, StateSchema} from '@langchain/langgraph';
import {ChatOpenAI} from '@langchain/openai'
import z from 'zod';

const run = async () => {
    const model = new ChatOpenAI('gpt-4o-mini');

    const schemaObject = {
        sentiment: z.enum(['positive', 'nigative']).describe('Sentiment of the review')
    }
    const SentimentsSchema = z.object(schemaObject);
    const structuredModel1 = model.withStructuredOutput(SentimentsSchema);

    const diagnosisSchemaObject = {
        issue_type: z.enum(['UX', 'Performance', 'Bug', 'Support', 'Other'])
            .describe('The category of issue mentioned in the review'),
        tone: z.enum(['angry', 'fustrated', 'disappointed', 'calm'])
            .describe('The emotional tone expressed by the user'),
        urgancy: z.enum(['low', 'medium', 'high'])
            .describe('How urgent or critical the issue appears to be')
    }
    const DiagnosisSchema = z.object(diagnosisSchemaObject);
    const structuredModel2 = model.withStructuredOutput(DiagnosisSchema);

    const reviewStateObj = {
        review: z.string(),
        sentiment: z.enum(['positive', 'negative']),
        diagnosis: z.object(),
        response: z.string()
    }
    const ReviewStateSchema = z.object(reviewStateObj);
    
    const findSentiment = async (state: z.infer<typeof ReviewStateSchema>) => {
        const prompt = ``;
        const sentiment = (await structuredModel1.invoke(prompt)).sentiment
        return {sentiment}
    }

    const checkSentiment = (state: z.infer<typeof ReviewStateSchema>): 'positive_response'|'run_diagnosis' => {
        if(state.sentiment === 'positive') {
            return 'positive_response'
        }
        return 'run_diagnosis';
    }

    const positiveResponse = async (state: z.infer<typeof ReviewStateSchema>) => {
        const prompt = `Write a warm thank-you message in response to this review:
                        \n\n\"${state.review}\"\n
                        Also, kindly ask the user to leave feedback on our website.`
        const response = (await model.invoke(prompt)).content;
        return {response};
    }

    const runDiagnosis = async (state: z.infer<typeof ReviewStateSchema>) => {
        const prompt = `Diagnose this negative review:\n\n ${state.review}\n" "Return issue_type, tone, and urgency`;
        const response = await structuredModel2.invoke(prompt);
        return {'diagnosis': response};
    }

    const negativeResponse = async (state: z.infer<typeof ReviewStateSchema>) => {
        const diagnosis = state.diagnosis;
        const prompt = `You are a support assistant.
            The user had a '${diagnosis.issue_type}' issue, sounded ${diagnosis.tone}', and marked urgency as '${diagnosis.urgency}'.
            Write an empathetic, helpful resolution message.`
        const response = (await model.invoke(prompt)).content;
        return {response};
    }

    const graph = new StateGraph(ReviewStateSchema);
    graph.addNode('find_sentiment', findSentiment)
        .addNode('positive_response', positiveResponse)
        .addNode('run_diagnosis', runDiagnosis)
        .addNode('negative_response', negativeResponse)
        
        .addEdge(START, 'find_sentiment')

        .addConditionalEdges('find_sentiment', checkSentiment) // checkSentiment returns 'positive_response' OR 'run_diagnosis'

        .addEdge('positive_response', END)

        .addEdge('run_diagnosis', 'negative_response')
        .addEdge('negative_response', END);
    
    const workflow = graph.compile();

    // positive input review
    // const result = await workflow.invoke({review: 'I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality.'});
    
    const result = await workflow.invoke({review: 'I am trying to login into system from last 2hr. I am getting blue screen. I have some important meetings scheduled. fix this issue as soon as possible.'});
    console.log(result);

}

export default run;