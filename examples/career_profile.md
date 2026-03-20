# Jordan Lee
jordan.lee@example.com | 415-555-0100 | San Francisco, CA
LinkedIn: linkedin.com/in/jordanlee

## Professional Summary
I'm an engineer who tends to end up at the intersection of distributed systems and team
leadership. I like problems where the technical and organisational challenges are equally hard.

## Work Experience

### Senior Software Engineer @ DataFlow Inc (Mar 2021 – Present)
DataFlow processes telemetry data for SaaS products. I joined as the team was planning a
platform overhaul.

**What I actually did:**
- Owned the architecture for our event ingestion service, which went from 2M to 10M events/day
  during my tenure without a reliability incident
- Led a 14-month migration from a Rails monolith to a Python-based microservices architecture;
  reduced average deploy time from 45 minutes to 18 minutes
- Reduced infrastructure spend by ~$180K/year by redesigning our Redis caching strategy —
  the win came from noticing that 40% of cache misses were for data we were computing identically
  in two separate services
- Mentored three junior engineers; two were promoted to mid-level within 18 months
- Introduced incident retrospectives — the team had never done them before; we've had zero
  repeat incident causes in the 18 months since

**Technologies:** Python, FastAPI, Kafka, PostgreSQL, Redis, AWS (ECS, RDS, ElastiCache), Terraform

---

### Software Engineer @ StartupXYZ (Jun 2019 – Feb 2021)
StartupXYZ was a Series A B2B SaaS company (analytics tooling). Small team, everything was fast.

**What I actually did:**
- Built the core REST API from scratch using Flask, then migrated it to FastAPI when we needed
  async support
- Wrote the first automated test suite the company had; got coverage to 87% in 3 months
- On-call for production incidents from month 2; learned a lot about what monitoring actually
  needs to catch

**Technologies:** Python, Flask/FastAPI, PostgreSQL, Docker, AWS

---

## Education

### B.S. Computer Science — UC Berkeley (2019)
- GPA: 3.7
- Relevant coursework: Distributed Systems, Database Systems, Operating Systems, Algorithms
- Honours: Dean's List (2017, 2018)

---

## Projects

### Open Source: kafka-retry (2023)
**Context:** There was no clean Python library for configurable Kafka retry-with-backoff that
worked well with our consumer group setup.
**My role:** Built it from scratch, published to PyPI.
**Outcome:** ~800 monthly downloads; two external contributors. We use it internally at DataFlow.
**Technologies:** Python, Kafka, pytest

### Internal Tool: Deploy Dashboard (2022)
**Context:** Engineers couldn't see the status of a deploy in progress without digging through
CloudWatch logs.
**My role:** Built a lightweight FastAPI + React dashboard that surfaces deploy state in real time.
**Outcome:** Used daily by 12 engineers; reduced "is the deploy done?" Slack messages to near zero.
**Technologies:** Python, FastAPI, React, WebSockets, AWS ECS

---

## Certifications
- AWS Certified Solutions Architect – Associate (2022)

---

## Key Points to Draw From
These are things I think are most relevant to staff+ roles:
- I've owned end-to-end architecture decisions, not just implementation
- I've made decisions that saved real money (the Redis caching work is the clearest example)
- I care about team velocity as much as technical correctness — the incident retrospective
  and deploy dashboard stories show this
- I have opinions about what makes distributed systems actually maintainable, not just scalable
- I've mentored people and it's gone well — I can give specific examples
- I'm good at the translation work between technical and non-technical stakeholders
