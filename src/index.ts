import dotenv from 'dotenv';
dotenv.config();

import helloWorld from "./hello-world";
import BMICal from "./SequentialWorkflow/bmi-cal-workflow";
import llmWorkflow from "./SequentialWorkflow/llm-workflow";
import promptChain from "./SequentialWorkflow/prompt-chaining";

// helloWorld();
// BMICal();
// llmWorkflow();
promptChain();

