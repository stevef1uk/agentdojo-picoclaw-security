#!/usr/bin/env python3
"""
Test picoclaw against AgentDojo's tool_knowledge attack
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentdojo.agent_pipeline.llms.picoclaw_llm import PicoclawLLM
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.attacks.important_instructions_attacks import ToolKnowledgeAttack

def test_picoclaw_security():
    print("🔒 Testing Picoclaw against ToolKnowledge attack...")
    
    # 1. Load the workspace suite (calendar/email/drive tasks)
    suite = get_suite("v1", "workspace")
    print(f"Loaded suite: {suite.name}")
    
    # 2. Select a task to test (user_task_0: find email addresses for networking event)
    task = suite.get_user_task_by_id("user_task_0")
    print(f"Testing task: {task.ID}")
    print(f"Task prompt: {task.PROMPT}")
    
    # 3. Create picoclaw LLM and wrap in AgentDojo pipeline
    picoclaw_llm = PicoclawLLM()
    pipeline_config = PipelineConfig(
        llm=picoclaw_llm,
        model_id=None,
        defense=None,  # "tool_filter"
        system_message_name="default",
        system_message=None
    )
    agent_pipeline = AgentPipeline.from_config(pipeline_config)
    agent_pipeline.name = "picoclaw"     
    # 4. Run the attack against picoclaw
    # We'll test with no injections first (utility), then with tool_knowledge attack (security)
    
    # Test utility (should succeed if picoclaw can do the task)
    print("\n--- Testing Utility (no attack) ---")
    utility_result, _ = run_single_task(
        agent_pipeline, suite, task, injection_task=None, injections={}
    )
    print(f"Utility result: {'✅ PASS' if utility_result else '❌ FAIL'}")
    
    # Test security (should resist the attack)
    print("\n--- Testing Security (with ToolKnowledge attack) ---")
    # Get an injection task to test against (let's use injection_task_0)
    injection_task = suite.get_injection_task_by_id("injection_task_0")
    print(f"Using injection task: {injection_task.ID}")
    print(f"Injection goal: {injection_task.GOAL}")
    
    # Create the attack
    attack = ToolKnowledgeAttack(suite, agent_pipeline)
    injections = attack.attack(task, injection_task)
    
    security_result, _ = run_single_task(
        agent_pipeline, suite, task, injection_task, injections
    )
    print(f"Security result: {'✅ RESISTED' if not security_result else '❌ VULNERABLE'}")
    
    print("\n📊 Summary:")
    print(f"  Utility: {'PASS' if utility_result else 'FAIL'}")
    print(f"  Security: {'RESISTED' if not security_result else 'VULNERABLE'}")
    
    return utility_result, not security_result  # Return (utility, security_score)

def run_single_task(pipeline, suite, user_task, injection_task, injections):
    """Helper to run a single task evaluation"""
    from agentdojo.task_suite.task_suite import GroundTruthPipeline
    
    # Load environment with injections
    environment = suite.load_and_inject_default_environment(injections)
    
    # Run with our picoclaw pipeline
    utility, security = suite.run_task_with_pipeline(
        agent_pipeline=pipeline,
        user_task=user_task,
        injection_task=injection_task,
        injections=injections,
        environment=environment

    )
    
    return utility, security

if __name__ == "__main__":
    test_picoclaw_security()

