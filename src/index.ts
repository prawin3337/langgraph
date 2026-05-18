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
tweetGenerator();
