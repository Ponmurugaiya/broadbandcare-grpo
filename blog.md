# From Chatbot to Thinking Agent: How I Built BroadbandCare with Reinforcement Learning

There's a moment in every side project where you stop and think, *"Wait — is this actually working?"*

For me, that moment came at 2 AM when I watched my small language model — a 1.5B parameter Qwen model — correctly diagnose a simulated fiber outage, escalate it to the right tier, and suggest a resolution, all by itself. No hardcoded rules. No lookup tables. Just a model that had learned, through trial and error, what a *good* broadband support agent actually does.

That's the story of **BroadbandCare** — and I want to walk you through it, honestly, including the parts where things broke spectacularly.

---

## Why I Started This

I've been fascinated by the gap between what LLMs *can* do and what they actually do in production support scenarios. Most chatbots are glorified FAQ machines. They match keywords, return a template response, and call it a day. If you've ever called your ISP and been routed to a bot that just kept apologizing — you know exactly what I mean.

I wanted to build something different. An agent that could *reason* through a problem: check the right tools, ask the right questions, and actually solve the issue rather than just deflecting.

The catch? I didn't want to rely on a massive 70B model. I wanted something small, fast, and trainable — something that could get *better* through experience.

That led me to **GRPO (Group Relative Policy Optimization)** and reinforcement learning from environment feedback.

---

## The Architecture: Teaching a Model to Think

At its core, BroadbandCare is a **Reinforcement Learning environment** where an LLM acts as an agent navigating telecom support cases.

Here's how it works:

### The Environment

I built a custom OpenEnv-compatible environment (`BroadbandCareEnv`) that simulates realistic broadband support scenarios — everything from simple modem reboots to complex fiber line diagnostics and billing escalations.

Each episode gives the agent:
- A customer complaint (the observation)
- A set of diagnostic tools it can call (ping test, line quality check, account lookup, escalation handler)
- A reward signal based on how well it resolves the issue

The **reward logic** was deterministic and carefully designed. The model got rewarded for:
- Using the right tool at the right time
- Asking clarifying questions before jumping to conclusions
- Correctly resolving the issue within a reasonable number of steps
- Formatting responses in a structured, professional way

And it got penalized for:
- Hallucinating tool outputs
- Giving irrelevant responses
- Exceeding step limits without resolution

### The Dataset

I generated **50–100 diverse telecom support cases** — each one crafted to test a different diagnostic skill. Slow speeds, intermittent dropouts, complete outages, router issues, billing disputes, modem compatibility problems — the full spectrum of what a real support agent faces daily.

Getting the dataset right was probably 40% of the work. Garbage in, garbage out applies doubly in RL.

---

## The Training Stack: Where Things Got Interesting

I used **Unsloth + TRL** for training — a combination that makes fine-tuning small models surprisingly fast, even on consumer hardware.

The base model: `Qwen2.5-1.5B-Instruct`. Small enough to iterate quickly, capable enough to understand multi-step reasoning.

The training pipeline:
1. **SFT (Supervised Fine-Tuning)** — First, I showed the model what good responses looked like. This gave it a baseline understanding of the task.
2. **GRPO Training** — Then, I let it loose in the environment and trained it with group relative policy optimization, rewarding better behaviors over time.

### The Dependency Hell Nobody Warns You About

I'm going to be real here: getting `trl`, `unsloth`, and `datasets` to all agree on compatible versions was *genuinely painful*. I spent an embarrassing amount of time debugging import errors and CUDA mismatches before landing on a stable configuration.

The fix? Pinning specific versions, isolating the training cell to be fully self-contained, and — most importantly — not updating anything once it works.

*If it works, don't touch it.* That's the unofficial first law of ML engineering.

---

## The Results: Did It Actually Learn?

Yes. And watching it happen was one of the most satisfying moments I've had building something.

I plotted reward curves across training steps and the improvement was visible — the model went from erratic, often irrelevant responses to consistently structured, tool-appropriate, resolution-focused answers.

Comparing **Base vs SFT vs GRPO**:
- **Base model**: Conversational but unfocused. Tends to apologize and deflect.
- **SFT model**: More structured, follows the format, but still mechanical.
- **GRPO model**: Actively *reasons*. Chooses tools based on context. Asks follow-up questions when the issue is ambiguous.

The jump from SFT to GRPO was the one that really mattered. That's where the agent stopped *performing* the role and started *thinking through* it.

---

## What I Learned

A few honest takeaways:

**1. Reward design is everything.**
The quality of your reward function determines the quality of your agent. I spent more time thinking about *what to reward* than almost anything else. Vague rewards produce vague agents.

**2. Small models can surprise you.**
1.5B parameters sounds tiny. But with the right training signal, even a small model can develop surprisingly coherent multi-step reasoning. You don't always need scale — you need the right incentives.

**3. Humanizing the data matters.**
My early dataset was too clean, too perfect. Real support conversations are messy — customers are frustrated, descriptions are vague, problems are ambiguous. Adding that noise to the training data made the agent dramatically more robust.

**4. RL is slow to debug.**
With supervised learning, you get an error and fix it. With RL, you get a reward signal that slowly drifts in a direction, and you have to *infer* what's causing it. Patience is non-negotiable.

---

## What's Next

BroadbandCare is currently deployed as a Hugging Face Space — a live demo where you can throw a broadband complaint at the agent and watch it reason through a diagnosis in real time.

The next steps I'm thinking about:
- **Multi-turn memory**: Right now, each turn is somewhat stateless. Adding conversation memory would make it dramatically more capable.
- **Human-in-the-loop feedback**: Using real support agent ratings to further refine the reward model.
- **Scaling up**: Testing whether GRPO gains hold when we move to a 7B model.

---

## Final Thoughts

Building BroadbandCare reminded me why I got into this field in the first place — the feeling of watching a system learn something you didn't explicitly program it to do.

Reinforcement learning is hard, messy, and often humbling. But when it clicks, there's nothing quite like it.

If you're thinking about building your own RL environment for a domain-specific agent, my honest advice: start smaller than you think you need to, obsess over your reward function, and don't underestimate 1.5B parameters with the right training signal.

The chatbot era is ending. The *thinking agent* era is just getting started.

---

*Built with Qwen2.5-1.5B, Unsloth, TRL, and too much coffee. ☕*

*[GitHub](https://github.com/Ponmurugaiya/broadbandcare-grpo)