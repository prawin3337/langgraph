import dotenv from 'dotenv';
dotenv.config();

import helloWorld from "./hello-world";
import BMICal from "./SequentialWorkflow/bmi-cal-workflow";
import llmWorkflow from "./SequentialWorkflow/llm-workflow";
import promptChain from "./SequentialWorkflow/prompt-chaining";
import batsmanWorkflow from './ParallelWorkflow/batsman-workflow';
import essayWorkflow from './ParallelWorkflow/essay-workflow';
import reviewReplyWorkflow from './ConditionalWorkflow/review-reply-workflow';
import tweetGenerator from './IterativeWorkflows/tweet-generator-workflow';
import chatbot from './Persistance/chatbot';
import streaming from './Streaming/streaming';
import tools from './Tools/chatbot';
import interrupt from './HumanInTheLoop/interrupt';
import hitl from './HumanInTheLoop/hitl';
// import chatbotHitl from './HumanInTheLoop/chatbot-hitl';
// import subgraphs from './Subgrapth/subgraphs';
import subgraphsSharedState from './Subgrapth/subgraphs_shared_state';


// helloWorld();

// Sequential Workflow
// BMICal();
// llmWorkflow();
// promptChain();

// Parallel Workflow
// batsmanWorkflow();
// essayWorkflow();

// Conditional Workflow
// reviewReplyWorkflow();

// Iterative Workflow
// tweetGenerator();

// chatbot();

// streaming();

// tools();

// interrupt
// hitl();
// chatbotHitl();

// subgraphs();
subgraphsSharedState();
