#!/usr/bin/env python3
import json, os, sys
last = ""
model = {"provider":"fake","id":"planner","name":"Planner","reasoning":True,"contextWindow":100000,"maxTokens":10000}
stats = {"sessionFile":"/tmp/fake.jsonl","sessionId":"fake","tokens":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0},"cost":0,"contextUsage":{"tokens":0,"contextWindow":100000,"percent":0}}
for raw in sys.stdin.buffer:
    req = json.loads(raw)
    rid = req.get("id")
    t = req.get("type")
    def resp(command, data=None, success=True, error=None):
        x={"id":rid,"type":"response","command":command,"success":success}
        if data is not None:x["data"]=data
        if error:x["error"]=error
        print(json.dumps(x),flush=True)
    if t=="get_available_models": resp(t,{"models":[model,{"provider":"fake","id":"worker","name":"Worker","reasoning":True,"contextWindow":100000,"maxTokens":10000}]})
    elif t=="set_model": model={**model,"provider":req["provider"],"id":req["modelId"]}; resp(t,model)
    elif t=="set_thinking_level": resp(t)
    elif t=="get_available_thinking_levels": resp(t,{"levels":["off","minimal","low","medium","high"]})
    elif t=="get_state": resp(t,{"model":model,"thinkingLevel":"high","sessionId":"fake","isStreaming":False})
    elif t=="get_session_stats": resp(t,stats)
    elif t=="get_last_assistant_text": resp(t,{"text":last})
    elif t=="prompt":
        msg=req.get("message","")
        if msg.startswith("/skill:"):
            last="READY"
        elif "Return STRICT JSON only" in msg and "hypotheses" in msg:
            last=json.dumps({"hypotheses":[{"title":"toy improvement","parent_ids":["baseline"],"hypothesis":"toy hypothesis","mechanism":"toy mechanism","expected_delta":0.2,"confidence":0.8,"novelty":0.7,"changes_hint":{"toy":[0,1]},"cheapest_falsification":"quick","difference_from_history":"new"}]})
        elif "scientific experiment reviewer" in msg:
            last=json.dumps({"status":"completed","lesson":"toy lesson","failure_reason":None,"confidence_update":0.8,"next_steps":[],"attribution":"toy","should_revisit":False,"belief_updates":[{"key":"toy","statement":"toy works","confidence":0.8}],"parameter_effects":[{"parameter":"toy","from":"0","to":"1","effect":0.1,"confidence":0.8}]})
        elif "candidate-selection critic" in msg:
            last=json.dumps({"selected":[{"index":0,"reason":"toy","priority":1}]})
        elif "meta-RSI reviewer" in msg:
            last=json.dumps({"rationale":"toy","policy":{"increase_budget":[],"decrease_budget":[],"revive_nodes":[],"crossovers":[],"new_directions":[],"exploration_ratio":0.3,"model_routing_notes":""}})
        elif "implementation worker" in msg:
            last="implemented toy change"
            if os.environ.get("FAKE_PI_ESCAPE"):
                # Simulates an agent that edits the main checkout instead of its worktree.
                try:
                    with open(os.environ["FAKE_PI_ESCAPE"], "a") as f:
                        f.write("# escaped edit\n")
                except OSError:
                    last="escape blocked by sandbox"
        else:
            last='{"ok": true}'
        resp(t,{"disposition":"started"})
        print(json.dumps({"type":"agent_start"}),flush=True)
        if "LOOP_TOOLS" in msg:
            # Simulates a runaway agent: endless tool calls, never settles until aborted.
            for i in range(1000):
                print(json.dumps({"type":"tool_execution_start","toolCallId":f"c{i}","toolName":"bash","args":{"command":"ls"}}),flush=True)
            continue
        if "BIG_LINE" in msg:
            last = "x" * 200_000  # larger than asyncio's 64 KiB default line limit
        if "HANG" in msg or ("implementation worker" in msg and os.environ.get("FAKE_PI_WORKER_HANG")):
            continue  # never settles; only abort ends it
        print(json.dumps({"type":"message_update","assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":last}}),flush=True)
        print(json.dumps({"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":last}],"provider":model["provider"],"model":model["id"]}}),flush=True)
        print(json.dumps({"type":"agent_settled"}),flush=True)
    elif t=="abort":
        print(json.dumps({"type":"agent_settled"}),flush=True)
        resp(t)
    elif t=="compact": resp(t,{"summary":"fake","tokensBefore":100,"estimatedTokensAfter":20})
    else: resp(t,success=False,error="unsupported")
